# Project Notes: Nepali NLP testing (A Grammarly-Inspired Approach)

## Overview
This project aims to improve a language model (LLM) specifically for Nepali. While current LLMs have been trained on Nepali data, enhancing their language-specific capabilities could make them more effective for Nepali users. Further, we also hope to see its applications on Old Nepali where ancient scripts may be involved.

## Project Goals
- Improving language modelling capabilities for Nepali
- Working with and build on top of the Indic Language models (IndicBERT, IndicBART, MuRIL)
- Work on tasks like (token classification + NER) OR (zero-shot classification - unsupervised) OR (feature extraction) OR (machine translations: checking translations old nepali to modern nepali)
- Additionally, as the next step see its applications on old nepali texts and scriptures (more reading required)

## Key Features
- Accumulating everything we know about Nepali LLMs so far, and see if we can make anything better
- There are many language models built for indian languages today but focusing on one specific language (Nepali) would help us pick up the nuances better
- Task specific focus
- Enhance fluency and coherence in Nepali sentences

## Tasks & To-Do List
- [ ] Model Testing
- [ ] Fine-tuning
- [ ] Sample Testing
- [ ] Pipeline development
- [ ] Final pipeline testing
- [ ] Fine tune it on old nepali (possibility of creating synthetic data) 
- [ ] Old Nepali testing

## Possible Pipeline (yet to finalize) 
- [step 1] Start with an indicBART to make the model understand the language better (sequence-to-sequence)
