"""Extrai o texto dos slides de um video MP4 de apresentacao via OCR.

O audio e completamente ignorado: o video e amostrado em intervalos regulares,
quadros duplicados/semelhantes sao descartados (perceptual hash) e o texto de
cada slide e extraido com Tesseract OCR, sendo estruturado em Markdown com base
na hierarquia visual (altura das linhas de texto no slide).

Instalacao das dependencias
---------------------------
1. Pacotes Python (dentro do virtualenv do projeto):

       pip install opencv-python pytesseract numpy
       # ou: pip install -r requirements.txt

2. Motor do Tesseract OCR (dependencia EXTERNA, obrigatoria para o pytesseract):

   - Windows:  winget install UB-Mannheim.TesseractOCR
     (instalador alternativo: https://github.com/UB-Mannheim/tesseract/wiki)
     Apos instalar, garanta que `tesseract.exe` esteja no PATH ou informe o
     caminho via variavel de ambiente TESSERACT_CMD, ex.:
     TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
   - Linux (Debian/Ubuntu):  sudo apt install tesseract-ocr tesseract-ocr-por
   - macOS:  brew install tesseract tesseract-lang

3. Para OCR em portugues, instale o pacote de idioma "por" (no instalador do
   Windows, marque "Portuguese" em Additional language data). Sem permissao de
   administrador, baixe `por.traineddata` de
   https://github.com/tesseract-ocr/tessdata_fast, copie-o junto com
   `eng.traineddata`/`osd.traineddata` para uma pasta sua (ex.:
   %LOCALAPPDATA%\\tessdata) e aponte a variavel de ambiente TESSDATA_PREFIX
   para essa pasta.

Uso
---
    python slide_text_extractor.py <caminho_arquivo> <nome_arquivo> [opcoes]

Exemplo:
    python slide_text_extractor.py ./videos "apresentacao.mp4" --saida slides.md
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytesseract
from pytesseract import Output, TesseractNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Caminhos padrao do Tesseract no Windows, usados como fallback quando o
# binario nao esta no PATH nem em TESSERACT_CMD.
DEFAULT_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


def configure_tesseract() -> None:
    """Localiza o binario do Tesseract e a pasta de idiomas (tessdata).

    Ordem de resolucao do binario: TESSERACT_CMD -> PATH -> caminhos padrao de
    instalacao no Windows. Para os idiomas, se TESSDATA_PREFIX nao estiver
    definido mas existir %LOCALAPPDATA%\\tessdata com o pacote 'por', usa essa
    pasta (cenario de instalacao sem permissao de administrador).
    """
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if not tesseract_cmd:
        from shutil import which

        tesseract_cmd = which("tesseract")
    if not tesseract_cmd:
        tesseract_cmd = next(
            (path for path in DEFAULT_TESSERACT_PATHS if Path(path).is_file()), None
        )
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    if not os.getenv("TESSDATA_PREFIX"):
        user_tessdata = Path(os.path.expandvars(r"%LOCALAPPDATA%")) / "tessdata"
        if (user_tessdata / "por.traineddata").is_file():
            os.environ["TESSDATA_PREFIX"] = str(user_tessdata)


configure_tesseract()

# --- Parametros de amostragem e deduplicacao -------------------------------

# Intervalo (segundos) entre quadros amostrados. 1s e suficiente para slides.
DEFAULT_SAMPLE_INTERVAL_S = 1.0
# Tamanho do perceptual hash (dHash): 16 -> 16*16 = 256 bits.
HASH_SIZE = 16
# Distancia de Hamming maxima para considerar dois quadros "o mesmo slide".
STABLE_DISTANCE = 10
# Distancia minima em relacao ao ultimo slide salvo para considerar "slide novo".
NEW_SLIDE_DISTANCE = 25
# Similaridade textual acima da qual dois slides sao tratados como duplicados
# (cobre revelacao incremental de bullets: mantemos a versao mais completa).
TEXT_SIMILARITY_THRESHOLD = 0.75

# --- Parametros de OCR ------------------------------------------------------

DEFAULT_OCR_LANGUAGES = "por+eng"
MIN_WORD_CONFIDENCE = 40   # confianca minima (0-100) por palavra do Tesseract
MIN_LINE_LENGTH = 3        # linhas mais curtas que isso sao descartadas (ruido)

# Marcadores tipicos de itens de lista em slides.
BULLET_PREFIX_RE = re.compile(r"^[\-\*\u2022\u25CF\u25AA\u25B6\u2023\u00BB>]\s+")


@dataclass
class OcrLine:
    """Uma linha de texto reconhecida no slide, com metadados de posicao."""

    text: str
    height: float  # altura media das palavras (proxy do tamanho da fonte)
    top: int       # posicao vertical no quadro (px)


@dataclass
class Slide:
    """Slide unico detectado no video, ja com o texto estruturado."""

    timestamp_s: float
    lines: list[OcrLine]

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


# --- Deteccao de quadros unicos ---------------------------------------------


def compute_dhash(frame: np.ndarray, hash_size: int = HASH_SIZE) -> np.ndarray:
    """Calcula o perceptual hash (dHash) de um quadro.

    O dHash compara a intensidade de pixels vizinhos numa versao reduzida da
    imagem; e robusto a ruido de compressao e pequenas variacoes (ex.: webcam
    do apresentador num canto do slide).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).flatten()


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def iter_unique_frames(
    video_path: Path, sample_interval_s: float
) -> Iterator[tuple[float, np.ndarray]]:
    """Amostra o video em intervalos regulares e emite apenas slides novos.

    Um quadro so e emitido quando esta ESTAVEL (igual ao quadro amostrado
    anterior, ou seja, fora de uma transicao/animacao) e DIFERENTE do ultimo
    slide emitido. Isso evita capturar quadros no meio de uma transicao e
    descarta duplicatas do mesmo slide.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir o video '{video_path}'. "
            "Verifique se o arquivo e um MP4 valido e se o codec e suportado."
        )

    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, round(fps * sample_interval_s))
        logger.info(
            "[iter_unique_frames] fps=%.2f, total_frames=%d, passo=%d quadros",
            fps, total_frames, frame_step,
        )

        previous_hash: np.ndarray | None = None
        last_emitted_hash: np.ndarray | None = None
        frame_index = 0

        while True:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                break

            current_hash = compute_dhash(frame)
            is_stable = (
                previous_hash is not None
                and hamming_distance(current_hash, previous_hash) <= STABLE_DISTANCE
            )
            is_new_slide = (
                last_emitted_hash is None
                or hamming_distance(current_hash, last_emitted_hash) >= NEW_SLIDE_DISTANCE
            )
            if is_stable and is_new_slide:
                timestamp_s = frame_index / fps
                last_emitted_hash = current_hash
                yield timestamp_s, frame

            previous_hash = current_hash
            frame_index += frame_step
    finally:
        capture.release()


# --- OCR e estruturacao do texto --------------------------------------------


def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    """Prepara o quadro para OCR: escala de cinza, upscale e inversao.

    Slides com fundo escuro sao invertidos, pois o Tesseract reconhece melhor
    texto escuro sobre fundo claro.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < 1600:  # upscale melhora OCR em videos de baixa resolucao
        scale = 1600 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if float(np.mean(gray)) < 127:
        gray = cv2.bitwise_not(gray)
    return gray


def extract_ocr_lines(frame: np.ndarray, languages: str) -> list[OcrLine]:
    """Roda o OCR no quadro e agrupa as palavras em linhas com posicao/altura."""
    image = preprocess_for_ocr(frame)
    data = pytesseract.image_to_data(image, lang=languages, output_type=Output.DICT)

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        confidence = float(data["conf"][i])
        if confidence < MIN_WORD_CONFIDENCE or not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines: list[OcrLine] = []
    for indexes in grouped.values():
        text = " ".join(data["text"][i].strip() for i in indexes)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < MIN_LINE_LENGTH:
            continue
        height = float(np.mean([data["height"][i] for i in indexes]))
        top = int(min(data["top"][i] for i in indexes))
        lines.append(OcrLine(text=text, height=height, top=top))

    lines.sort(key=lambda line: line.top)
    return lines


def slide_to_markdown(slide: Slide, slide_number: int, frame_height: int) -> list[str]:
    """Converte as linhas OCR de um slide em Markdown hierarquico.

    Heuristica de hierarquia visual:
    - Titulo (#): linha de maior fonte localizada no terco superior do slide.
    - Subtopico (##): linhas com fonte acima da mediana, sem marcador de lista.
    - Itens (-): linhas com marcador de lista ou com fonte igual/abaixo da mediana.
    """
    if not slide.lines:
        return []

    heights = [line.height for line in slide.lines]
    median_height = float(np.median(heights))

    title_line: OcrLine | None = None
    top_region = [line for line in slide.lines if line.top < frame_height / 3]
    if top_region:
        candidate = max(top_region, key=lambda line: line.height)
        if candidate.height >= median_height * 1.05 or len(slide.lines) == 1:
            title_line = candidate

    minutes, seconds = divmod(int(slide.timestamp_s), 60)
    output: list[str] = []
    if title_line is not None:
        output.append(f"# {BULLET_PREFIX_RE.sub('', title_line.text)}")
    else:
        output.append(f"# Slide {slide_number} ({minutes:02d}:{seconds:02d})")

    for line in slide.lines:
        if line is title_line:
            continue
        clean = BULLET_PREFIX_RE.sub("", line.text)
        if BULLET_PREFIX_RE.match(line.text) or line.height <= median_height * 1.1:
            output.append(f"- {clean}")
        else:
            output.append(f"## {clean}")

    output.append("")  # linha em branco separando slides
    return output


def text_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a, text_b).ratio()


def deduplicate_slides(slides: list[Slide]) -> list[Slide]:
    """Remove slides com texto quase identico ao anterior.

    Cobre o caso de bullets revelados gradualmente: quando dois slides
    consecutivos sao muito parecidos, mantemos o que tem mais texto.
    """
    unique: list[Slide] = []
    for slide in slides:
        if unique and text_similarity(unique[-1].full_text, slide.full_text) >= TEXT_SIMILARITY_THRESHOLD:
            if len(slide.full_text) > len(unique[-1].full_text):
                unique[-1] = slide
            continue
        unique.append(slide)
    return unique


# --- Funcao principal --------------------------------------------------------


def extract_presentation_text(
    caminho_arquivo: str,
    nome_arquivo: str,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    languages: str = DEFAULT_OCR_LANGUAGES,
    output_file: str | None = None,
    max_duration_s: float | None = None,
) -> str:
    """Extrai o texto dos slides de um video MP4 e devolve Markdown estruturado.

    Args:
        caminho_arquivo: diretorio onde o video esta localizado.
        nome_arquivo: nome do arquivo MP4.
        sample_interval_s: intervalo de amostragem de quadros, em segundos.
        languages: idiomas do Tesseract (ex.: "por+eng").
        output_file: se informado, grava o Markdown nesse arquivo (UTF-8).
        max_duration_s: processa apenas os N segundos iniciais (util em testes).

    Returns:
        O conteudo Markdown com o texto de todos os slides detectados.

    Raises:
        NotADirectoryError: se o diretorio nao existir.
        FileNotFoundError: se o arquivo de video nao existir.
        RuntimeError: se o video nao puder ser aberto ou o Tesseract faltar.
    """
    directory = Path(caminho_arquivo)
    if not directory.is_dir():
        raise NotADirectoryError(f"Diretorio invalido: '{caminho_arquivo}'")

    video_path = directory / nome_arquivo
    if not video_path.is_file():
        raise FileNotFoundError(f"Arquivo de video nao encontrado: '{video_path}'")
    if video_path.suffix.lower() != ".mp4":
        logger.warning(
            "[extract_presentation_text] extensao inesperada '%s' (esperado .mp4); "
            "tentando processar mesmo assim", video_path.suffix,
        )

    logger.info("[extract_presentation_text] processando '%s'", video_path)

    slides: list[Slide] = []
    frame_height = 0
    try:
        for timestamp_s, frame in iter_unique_frames(video_path, sample_interval_s):
            if max_duration_s is not None and timestamp_s > max_duration_s:
                break
            frame_height = preprocess_for_ocr(frame).shape[0]
            lines = extract_ocr_lines(frame, languages)
            if not lines:
                continue
            slides.append(Slide(timestamp_s=timestamp_s, lines=lines))
            logger.info(
                "[extract_presentation_text] slide candidato em %.0fs (%d linhas)",
                timestamp_s, len(lines),
            )
    except TesseractNotFoundError as error:
        raise RuntimeError(
            "Motor Tesseract OCR nao encontrado. Instale-o (ver instrucoes no "
            "topo deste arquivo) e/ou defina a variavel de ambiente TESSERACT_CMD "
            "com o caminho do executavel."
        ) from error

    slides = deduplicate_slides(slides)
    logger.info("[extract_presentation_text] %d slides unicos detectados", len(slides))

    markdown_lines: list[str] = []
    for number, slide in enumerate(slides, start=1):
        markdown_lines.extend(slide_to_markdown(slide, number, frame_height))
    markdown = "\n".join(markdown_lines).strip() + "\n"

    if output_file:
        Path(output_file).write_text(markdown, encoding="utf-8")
        logger.info("[extract_presentation_text] resultado gravado em '%s'", output_file)

    return markdown


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extrai o texto dos slides de um video MP4 de apresentacao (OCR)."
    )
    parser.add_argument("caminho_arquivo", help="Diretorio onde o video esta localizado")
    parser.add_argument("nome_arquivo", help="Nome do arquivo de video MP4")
    parser.add_argument(
        "--saida",
        help="Arquivo Markdown de saida (padrao: <nome_do_video>_slides.md no diretorio do video)",
    )
    parser.add_argument(
        "--intervalo", type=float, default=DEFAULT_SAMPLE_INTERVAL_S,
        help="Intervalo de amostragem em segundos (padrao: %(default)s)",
    )
    parser.add_argument(
        "--idiomas", default=DEFAULT_OCR_LANGUAGES,
        help="Idiomas do Tesseract, ex.: por+eng (padrao: %(default)s)",
    )
    parser.add_argument(
        "--duracao-maxima", type=float, default=None,
        help="Processa apenas os N segundos iniciais do video (util para testes)",
    )
    args = parser.parse_args()

    output_file = args.saida or str(
        Path(args.caminho_arquivo) / f"{Path(args.nome_arquivo).stem}_slides.md"
    )

    try:
        markdown = extract_presentation_text(
            caminho_arquivo=args.caminho_arquivo,
            nome_arquivo=args.nome_arquivo,
            sample_interval_s=args.intervalo,
            languages=args.idiomas,
            output_file=output_file,
            max_duration_s=args.duracao_maxima,
        )
    except (NotADirectoryError, FileNotFoundError, RuntimeError) as error:
        logger.error("[main] %s", error)
        return 1

    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
