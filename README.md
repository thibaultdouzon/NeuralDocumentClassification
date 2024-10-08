# Neural Document Classification

Document image classification with neural networks on a subset of the RVL-CDIP dataset [[1]](#1).

## Getting Started

The classification problem is tackled with three different approaches:

* Visual approach over the image pixels with dense only and convolutional neural networks: ```chapter_1_vision.ipynb```
* Textual approach over the recognized image words with bag-of-words, word embedding models and pre-trained transformers: ```chapter_2_text.ipynb```
* Lazy approach, using a multimodal LLM to perform the classfication without any finetuning. Time to practice prompt engineering techniques: ```chapter_3_vllm.ipynb```

It is recommended to begin with the visual approach as it includes more details about the computing environment setup and the dataset.

For a better experience, execute the notebooks within a Google Colab environment.

## Authors

* **Thibault Douzon** - [thibaultdouzon](https://github.com/thibaultdouzon)
* **Jérémy Espinas**
* **Clément Sage** - [clemsage](https://github.com/clemsage)
* **Bertrand Buffat** - [berbuf](https://github.com/berbuf)

## References

<a id="1">[1]</a> A. W. Harley, A. Ufkes, K. G. Derpanis, "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," in ICDAR, 2015
