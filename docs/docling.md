# Docling v2 Overview

## What's New
Docling v2 introduces several new features:
- Supports PDF, MS Word, MS PowerPoint, HTML, and various image formats.
- Provides a universal document representation encapsulating document hierarchy.
- Features a new API and CLI.

## Changes in Docling v2


### DocumentConverter Setup
- The setup for `DocumentConverter` has changed to accommodate many input formats.
  
  **Example:**
  ```python
  from docling.document_converter import DocumentConverter
  from docling.datamodel.base_models import InputFormat
  from docling.document_converter import PdfFormatOption, WordFormatOption
  from docling.pipeline.simple_pipeline import SimplePipeline
  from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
  from docling.datamodel.pipeline_options import PdfPipelineOptions
  from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

  pipeline_options = PdfPipelineOptions()
  pipeline_options.do_ocr = False
  pipeline_options.do_table_structure = True

  doc_converter = DocumentConverter(
      allowed_formats=[
          InputFormat.PDF,
          InputFormat.IMAGE,
          InputFormat.DOCX,
          InputFormat.HTML,
          InputFormat.PPTX,
      ],
      format_options={
          InputFormat.PDF: PdfFormatOption(
              pipeline_options=pipeline_options,
              backend=PyPdfiumDocumentBackend
          ),
          InputFormat.DOCX: WordFormatOption(
              pipeline_cls=SimplePipeline
          ),
      },
  )
  ```

### Document Conversion
- Simplified methods for converting documents.
  
  **Example:**
  ```python
  from docling.datamodel.document import ConversionResult

  # Convert a single file
  conv_result: ConversionResult = doc_converter.convert("https://arxiv.org/pdf/2408.09869")

  # Convert several files at once
  input_files = [
      "tests/data/wiki_duck.html",
      "tests/data/word_sample.docx",
      "tests/data/lorem_ipsum.docx",
      "tests/data/powerpoint_sample.pptx",
      "tests/data/2305.03393v1-pg9-img.png",
      "tests/data/2206.01062.pdf",
  ]
  conv_results_iter = doc_converter.convert_all(input_files)
  ```

### Accessing Document Structures
- Access and export converted document data using the `DoclingDocument` object.

  **Example:**
  ```python
  conv_result: ConversionResult = doc_converter.convert("https://arxiv.org/pdf/2408.09869")
  conv_result.document.print_element_tree()

  for item, level in conv_result.document.iterate_items:
      if isinstance(item, TextItem):
          print(item.text)
      elif isinstance(item, TableItem):
          table_df: pd.DataFrame = item.export_to_dataframe()
          print(table_df.to_markdown())
  ```

### Exporting Formats
- New methods for exporting to JSON, Markdown, and Doctags.

  **Example:**
  ```python
  print(json.dumps(conv_result.document.export_to_dict()))
  print(conv_result.document.export_to_markdown())
  print(conv_result.document.export_to_document_tokens())
  ```

### Reloading Documents
- Save and reload `DoclingDocument` in JSON format.

  **Example:**
  ```python
  # Save to disk
  with Path("./doc.json").open("w") as fp:
      fp.write(json.dumps(doc.export_to_dict()))

  # Load from disk
  with Path("./doc.json").open("r") as fp:
      doc_dict = json.loads(fp.read())
      doc = DoclingDocument.model_validate(doc_dict)
  ```

