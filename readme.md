# ISCOMatcher: The Smart AI Solution for Job Classification

**ISCOMatcher** is a lightweight AI tool designed for fast, secure, and training-free ISCO job classification.

ISCOMatcher leverages advanced AI to automate job classification using the ISCO system, efficiently identifying and ranking the most similar job roles. The tool is optimized to handle messy, lengthy text by cleaning and extracting only the essential information required for accurate classification. It can manage very long texts without any limit, thanks to techniques like sliding windows, which break down the text into manageable chunks for precise analysis. Its lightweight architecture allows it to run locally on small machines, making it ideal for use with sensitive data.

Built with flexibility in mind, ISCOMatcher is a generic solution that can be applied to any classification task. Simply provide a classification description, and it will adapt to your specific needs. Whether you're dealing with job data or other types of information, ISCOMatcher provides a fast, scalable, and secure classification solution with no extensive training required.

## Classes
The file data_description.csv specifies the number of classes, which in this case is 436.

## Features

- **Lightweight & Secure**: Can run locally, making it suitable for sensitive data.
- **Messy Text Handling**: Cleans and processes unstructured or lengthy text for precise classification.
- **Handles Long Text**: No text length limitations due to sliding window techniques that divide text into manageable parts.
- **Generic Solution**: Easily adaptable for any classification task—just provide a classification description.
- **Multilingual Support**: Handles data in multiple languages without additional training.

## Components

### Language Detection
- **langdetect**: Supports 55 languages.  
  [Learn More](https://pypi.org/project/langdetect/)
- In this project, I aimed to process the following 27 languages (ISO 639-1 codes): en, pl, de, it, bg, ro, sl, pt, et, hu, nl, lt, sv, fr, sk, cs, lv, es, el, fi, hr, da, ru, tl, ca, vi, af.
- If you are interested in other languages, please check for compatibility.

### Translation
- **M2M100 (facebook/m2m100_1.2B)**: A multilingual seq-to-seq model capable of direct translation between 100 languages (9,900 directions).  
  [Learn More](https://huggingface.co/facebook/m2m100_1.2B)

### Text Cleaning
- **BART (Large) – Fine-Tuned on CNN Daily Mail**: A seq2seq transformer with a BERT-like encoder and GPT-like decoder. It excels in text summarization and generation, pre-trained by corrupting text and learning to reconstruct it.  
  [Learn More](https://huggingface.co/facebook/bart-large-cnn)

### Embeddings
- **all-mpnet-base-v2**: A sentence-transformer model that maps sentences and paragraphs into a 768-dimensional dense vector space.  
  [Learn More](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

### Classification / Information Retrieval
- **Faiss**: A library for efficient similarity search in large-scale datasets. It optimizes nearest-neighbor searches for millions to billions of vectors and supports both CPU and GPU implementations.  
  The output is the 5 most similar ISCO job roles, ranked by relevance.  
  [Learn More](https://github.com/facebookresearch/faiss)

## Processing Example Data

### Example Data of Query/OJA :
This is how the .csv/query looks like:
| ID | Title                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|----|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0  | "Softwareentwickler (m/w/d)" | "Wir suchen einen talentierten Softwareentwickler (m/w/d) zur Verstärkung unseres Teams. Du wirst an innovativen IT-Projekten arbeiten und maßgeschneiderte Softwarelösungen entwickeln. Aufgaben: Entwicklung und Optimierung von Softwarelösungen. Zusammenarbeit mit anderen Entwicklern und Designern. Durchführung von Tests und Fehleranalysen. Anforderungen: Abgeschlossenes Informatikstudium oder vergleichbare Qualifikation. Erfahrung mit Python, Java oder C++. Gute Kenntnisse in agilen Entwicklungsmethoden. Teamfähigkeit und selbstständige Arbeitsweise. Wir bieten: Attraktive Vergütung und flexible Arbeitszeiten. Homeoffice-Möglichkeiten. Bewerbung: Sende deine Bewerbung mit Lebenslauf an karriere@techvision.de." |


### Text-Based Processing Steps for Query/OJA:
This table illustrates how the Title and Description are preprocessed before being transformed into an embedding:
| Step                         | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Title (original)             | "Softwareentwickler (m/w/d)"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Description (original)       | "Wir suchen einen talentierten Softwareentwickler (m/w/d) zur Verstärkung unseres Teams. Du wirst an innovativen IT-Projekten arbeiten und maßgeschneiderte Softwarelösungen entwickeln. Aufgaben: Entwicklung und Optimierung von Softwarelösungen. Zusammenarbeit mit anderen Entwicklern und Designern. Durchführung von Tests und Fehleranalysen. Anforderungen: Abgeschlossenes Informatikstudium oder vergleichbare Qualifikation. Erfahrung mit Python, Java oder C++. Gute Kenntnisse in agilen Entwicklungsmethoden. Teamfähigkeit und selbstständige Arbeitsweise. Wir bieten: Attraktive Vergütung und flexible Arbeitszeiten. Homeoffice-Möglichkeiten. Bewerbung: Sende deine Bewerbung mit Lebenslauf an karriere@techvision.de." |
| Language Detection           | "de" (German)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
| Translation: Title           | "Software developer (m/w/d)"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
| Translation: Description     | "We are looking for a talented software developer (m/w/d) to strengthen our team. You will work on innovative IT projects and develop tailor-made software solutions. Tasks: development and optimization of software solutions. collaboration with other developers and designers. conducting tests and error analysis. Requirements: Completed computer science study or comparable qualification. experience with Python, Java or C++. Good knowledge in agile development methods. team skills and independent work. We offer: attractive remuneration and flexible working hours. Home office opportunities. Application: Apply with a resume at karriere@visiontech.de."                                                                  |
| Text Cleaning: Description   | "We are looking for a talented software developer (m/w/d) to strengthen our team. You will work on innovative IT projects and develop tailor-made software solutions. Tasks: development and optimization of software solutions, testing and error analysis."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

### Example Result
After Faiss searches for the most similar embeddings, the top N embeddings will be displayed:
| Rank | Title                                                 | isco_uri                              |
|------|-------------------------------------------------------|---------------------------------------|
| 1    | 97 Software developers                                | http://data.europa.eu/esco/isco/C2512 |
| 2    | 99 Applications programmers                           | http://data.europa.eu/esco/isco/C2514 |
| 3    | 98 Web and multimedia developers                      | http://data.europa.eu/esco/isco/C2513 |
| 4    | 100 Software and applications developers and analy... | http://data.europa.eu/esco/isco/C2519 |
| 5    | 96 Systems analysts                                   | http://data.europa.eu/esco/isco/C2511 |

🔗 [Read the story about the ISCOMatcher on Medium](https://medium.com/@jweissmann/iscomatcher-revolutionizing-multilingual-job-advertisement-classification-56d1abb364c4)