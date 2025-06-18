# metal-products-ner

NER system that automatically extracts 27 types of technical specifications from metal product descriptions using hybrid ML approaches.

## Key Features

Multi-standard recognition: TU, GOST, EN, DIN, and ISO standards

Composite value parsing: Extracts dimensions like "57×3 mm" → (diameter:57, thickness:3)

Context-aware classification: Distinguishes between "AISI 304" (steel grade) vs "304 mm" (width)

Robust to noise: Works with incomplete descriptions and typos

Production pipeline: Includes preprocessing and ML models.

## Performance

