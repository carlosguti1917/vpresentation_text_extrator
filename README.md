# vpresentation_text_extrator

Extrator de textos de apresentações em vídeos MP4. O vídeo é amostrado em
intervalos regulares, quadros duplicados/semelhantes são descartados
(perceptual hash) e o texto de cada slide é extraído via OCR (Tesseract),
gerando um Markdown estruturado por títulos, subtópicos e itens de lista.
A trilha de áudio é completamente ignorada.

## Instalação

```powershell
# 1. Dependências Python (dentro do virtualenv)
pip install -r requirements.txt

# 2. Motor Tesseract OCR (dependência externa)
winget install UB-Mannheim.TesseractOCR
```

Para OCR em português, o pacote de idioma `por` precisa estar no diretório
`tessdata` do Tesseract. Sem permissão de administrador, baixe
[`por.traineddata`](https://github.com/tesseract-ocr/tessdata_fast) para uma
pasta sua (ex.: `%LOCALAPPDATA%\tessdata`, junto com `eng.traineddata` e
`osd.traineddata` copiados da instalação) e defina as variáveis de ambiente:

```powershell
[Environment]::SetEnvironmentVariable('TESSERACT_CMD','C:\Program Files\Tesseract-OCR\tesseract.exe','User')
[Environment]::SetEnvironmentVariable('TESSDATA_PREFIX',"$env:LOCALAPPDATA\tessdata",'User')
```

## Uso

```powershell
python slide_text_extractor.py <caminho_arquivo> <nome_arquivo> [opções]

# Exemplo
python slide_text_extractor.py .\videos "apresentacao.mp4" --saida slides.md

# Opções
#   --saida ARQUIVO          arquivo Markdown de saída (padrão: <video>_slides.md)
#   --intervalo SEGUNDOS     intervalo de amostragem de quadros (padrão: 1.0)
#   --idiomas LANGS          idiomas do Tesseract (padrão: por+eng)
#   --duracao-maxima SEG     processa apenas os N segundos iniciais (testes)
```

O resultado segue a hierarquia:

```markdown
# [Título do slide]
## [Subtópico ou texto explicativo]
- [Itens de lista / detalhes]
```

Também pode ser usado como biblioteca:

```python
from slide_text_extractor import extract_presentation_text

markdown = extract_presentation_text("./videos", "apresentacao.mp4")
```
