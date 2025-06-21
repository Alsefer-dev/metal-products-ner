# metal-products-ner

NER system that automatically extracts 28 types of technical specifications from metal product descriptions using hybrid ML approaches.

## Key Features

### Data Composition

70% Manufacturer Data: High-quality structured descriptions

20% Marketplace Data: Natural noise for robustness

10% Synthetic Data: Controlled error injection

### Entity Distribution

28 entity types (e.g., product, thickness, standart_gost)

Imbalanced classes (product=18.05% vs package=0.10%)

### Technical Specs

Format: JSONL

Avg. tokens/description: 9-10

Entity density: 72% of tokens

### Benchmark Results

Best model: CNN-BiLSTM-LayerNorm (F1=0.97)

### Use Cases

Product catalog automation

Supply chain data extraction

Quality control documentation

All data collected from public sources with ethical compliance.

## Dataset

### Source data and common information

| Source Type                     | Data Quality | Sample Share | Primary Purpose                 | Key Characteristics                          |
|---------------------------------|--------------|--------------|----------------------------------|---------------------------------------------|
| **Manufacturer Catalogs & E-Shops** | High        | 70%          | Core training foundation         | Standardized formats, verified specifications |
| **Marketplaces**<br>(Avito, MetallBase) | Medium    | 20%          | Model robustness enhancement    | Natural noise (typos, incomplete entries)    |
| **Synthetic Data**<br>(LLM-generated) | Low       | 10%          | Error resistance training       | Controlled variations (+5% deliberate errors) |

### Train/test split

| Split	|Total tokens |	Entity tokens	| Entity %	|
|------|--------------|---------|----|
|Train	| 138 209	| 100 365	| 72.6% |
|Test	| 59 501 |	43 025	| 72.3%	|

### Entity type statistics

| Entity               | Type        | Description                              | Value Examples (RUS)          | Entity count  | % of Total |
|----------------------|-------------|------------------------------------------|-------------------------------|--------|------------|
| **coating**         | General     | Surface treatment/coating               | "оцинкованное", "полимерное"  | 654    | 0.79%      |
| **color**           | General     | Product color                           | "черный", "RAL 9005"          | 324    | 0.39%      |
| **country**         | General     | Manufacturing origin                    | "Россия", "Китай"             | 95     | 0.12%      |
| **form**            | Specialized | Geometric profile                       | "рифленый", "гладкий"         | 676    | 0.82%      |
| **height**          | Specialized | Product height (mm)                     | 50, 250                       | 1,899  | 2.30%      |
| **height_big**      | Specialized | Large flange height (angles)            | 100                           | 583    | 0.71%      |
| **height_small**    | Specialized | Small flange height (angles)            | 65                            | 43     | 0.05%      |
| **inner_diameter**  | Specialized | Pipe inner diameter (mm)                | 15                            | 4,906  | 5.94%      |
| **length**          | General     | Product length (mm/m)                   | 6000, 12                      | 4,082  | 4.94%      |
| **manufacturer**    | General     | Producer company                        | "Северсталь", "NLMK"          | 506    | 0.61%      |
| **mark**            | General     | Factory marking                         | -                             | 1,548  | 1.87%      |
| **mark_steel**      | General     | Steel grade (GOST)                      | "Ст3", "09Г2С"                | 9,190  | 11.13%     |
| **mark_steel_aisi** | General     | Steel grade (AISI/ISO)                  | "AISI 304", "S355JR"          | 1,800  | 2.18%      |
| **material**        | General     | Base metal composition                  | "нержавеющая сталь", "сталь"  | 6,501  | 7.87%      |
| **outer_diameter**  | Specialized | Pipe outer diameter (mm)                | 21.3                          | 2,088  | 2.53%      |
| **package**         | General     | Packaging method                        | "бухта", "паллет"             | 82     | 0.10%      |
| **precision**       | Specialized | Manufacturing tolerance class           | "A", "B", "h11", "±0.5mm"     | 140    | 0.17%      |
| **product**         | General     | Product name                            | "труба", "уголок"             | 14,904 | 18.05%     |
| **purpose**         | General     | Intended application                    | "для строительства"           | 115    | 0.14%      |
| **standart_en**     | General     | International standard                  | "EN 10025", "ASTM A53"        | 967    | 1.17%      |
| **standart_gost**   | General     | Russian standard (GOST)                 | "ГОСТ 5781-82"                | 5,372  | 6.51%      |
| **standart_tu**     | General     | Enterprise specifications               | "ТУ 14-1-5523-2006"           | 276    | 0.33%      |
| **strength_class**  | Specialized | Mechanical strength rating              | "А500С"                       | 369    | 0.45%      |
| **strength_class_old** | Specialized | Legacy strength class                | "А3"                          | 231    | 0.28%      |
| **tehnology**       | General     | Production method                       | "горячекатаный", "холоднодеформированный" | 3,617 | 4.38% |
| **thickness**       | Specialized | Material thickness (mm)                 | 0.5, 12                       | 9,198  | 11.14%     |
| **type**            | General     | Product sub-category                    | "электросварная", "калиброванный" | 6,265 | 7.59% |
| **width**           | Specialized | Product width measurement               | 1000, 1250                    | 6,143  | 7.44%      |
| **Total**           | -           | -                                       | -                             | 82,574 | 100%       |

## Baseline

| Architecture                          | Framework | Backbone Config                          | Precision | Recall | F1   | Train Time (CPU, s) | Inference Speed (tok/s) |
|---------------------------------------|-----------|------------------------------------------|-----------|--------|------|---------------------|-------------------------|
| **Baseline (Current SOTA)**           |           |                                          |           |        |      |                     |                         |
| CNN+BiLSTM+LayerNorm+Linear           | PyTorch   | 1CNN(k=3), 1BiLSTM, LayerNorm, Linear    | 0.98      | 0.95   | 0.97 | 429                 | 13,361                  |
| **CNN Variants**                      |           |                                          |           |        |      |                     |                         |
| 1CNN+Linear                           | PyTorch   | 1CNN(k=3), Linear                        | 0.91      | 0.85   | 0.87 | 62                  | 23,008                  |
| 2CNN+Linear                           | PyTorch   | 2CNN(k=3,k=5), Linear                    | 0.97      | 0.93   | 0.95 | 133                 | 19,827                  |
| 3CNN+Linear                           | PyTorch   | 3CNN(k=3,k=5,k=7), Linear                | 0.97      | 0.94   | 0.95 | 208                 | 16,548                  |
| 4CNN+Linear                           | PyTorch   | 4CNN(k=3,k=5,k=5,k=7), Linear            | 0.95      | 0.93   | 0.94 | 239                 | 15,257                  |
| 4CNN+CRF                              | spaCy     | -                                        | 0.97      | 0.93   | 0.95 | 1,800               | 10,405                  |
| **CNN+LSTM/BiLSTM**                   |           |                                          |           |        |      |                     |                         |
| CNN+LSTM+Linear                       | PyTorch   | 1CNN(k=3), 1LSTM, Linear                 | 0.98      | 0.94   | 0.96 | 187                 | 17,279                  |
| CNN+BiLSTM+Linear                     | PyTorch   | 1CNN(k=3), 1BiLSTM, Linear               | 0.97      | 0.95   | 0.96 | 239                 | 13,529                  |
| 2CNN+LSTM+Linear                      | PyTorch   | 2CNN(k=3,k=5), 1LSTM, Linear             | 0.97      | 0.94   | 0.95 | 362                 | 11,597                  |
| 2CNN+BiLSTM+Linear                    | PyTorch   | 2CNN(k=3,k=5), 1BiLSTM, Linear           | 0.96      | 0.94   | 0.95 | 475                 | 11,263                  |
| **CNN+BiLSTM+LayerNorm**              |           |                                          |           |        |      |                     |                         |
| CNN+BiLSTM+LayerNorm+Linear           | PyTorch   | 1CNN(k=3), 1BiLSTM, LayerNorm, Linear    | 0.98      | 0.95   | 0.97 | 429                 | 13,361                  |
| **Attention Variants**                |           |                                          |           |        |      |                     |                         |
| CNN+BiLSTM+Attention+LayerNorm+Linear | PyTorch   | 1CNN(k=3), 1BiLSTM, 4-head Attn, LayerNorm | 0.97    | 0.94   | 0.95 | 588                 | 9,353                   |
| BiLSTM+Attention+LayerNorm+Linear     | PyTorch   | 1BiLSTM, 4-head Attn, LayerNorm          | 0.97      | 0.95   | 0.96 | 727                 | 11,852                  |
| 2BiLSTM+Attention+LayerNorm+Linear    | PyTorch   | 2BiLSTM, 4-head Attn, LayerNorm          | 0.98      | 0.95   | 0.96 | 1,005               | 7,206                   |
| **Transformer**                       |           |                                          |           |        |      |                     |                         |
| Transformer                           | spaCy     | DeepPavlov/rubert-base-cased             | 0.96         | 0.94      | 0.95    | -                   | -                       |

**Key Findings**:
1. **Best F1**: CNN+BiLSTM+LayerNorm+Linear (0.97)
2. **Fastest Inference**: 1CNN+Linear (23,008 tok/s)
3. **Accuracy-Speed Tradeoff**: Adding attention reduces speed 2x but maintains ~0.95 F1
4. **LayerNorm Benefit**: +0.01 F1 over vanilla BiLSTM


