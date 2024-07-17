## CLIP-LogisticRegression for CIFAR-100

This repository contains a simple LogisticRegression model utilizing OpenAI's CLIP model, achieving approximately 85% accuracy on the CIFAR-100 dataset.

### Features

- **High Accuracy**: Achieves around 85% accuracy on CIFAR-100 using CLIP features.
- **Scalability**: Can be trained with different sizes of CLIP pretrained models for varying trade-offs between accuracy and training speed.
- **Versatility**: Adaptable for training on other datasets by leveraging CLIP's image features.
- **Ease of Use**: Includes a Jupyter notebook (`CLIP-CIFAR100.ipynb`) documenting the training process and a sample script (`test.py`) for straightforward execution.

### Getting Started

1. **Environment Setup**:
   - Ensure Python environment with necessary dependencies (All in file **requirements.txt**).
   Run command below:
       ```pip install -r requirements.txt```
2. **Training Process Details**:
   - Refer to `CLIP-CIFAR100.ipynb` for detailed training steps and configurations.

3. **Testing**:
   - Since the pre-trained models are already saved, you can just run the `test.py` file.
        ```python test.py``` 
   - You can edit the CLIP model name in the script to change between **ViT-B/32** and **ViT-L/14** models, you also need to change the linear model name accordingly.
   - **Advanced Usage**: Open `CLIP-CIFAR100.ipynb` file and edit the dataset and CLIP pretrained model, then train a new linear model yourself.

### Difference in Models
   All models are saved in **./models** directory, CLIP models should be downloaded automatically in the code.
   - ViT-L/14  0.8478(85%) 
   - ViT-B/32  0.7816(78%)
   Note: There's a visual evaluation file `./resources/visual_evaluation.png` containing sample predictions of model **ViT-L/14**.
### License

This project is licensed under the MIT License - see the `LICENSE` file for details.

### Contributions

Special thanks to OpenAI for their CLIP model and the CIFAR-100 dataset creators for providing benchmark data. Also using OpenAI's GPT-4o for code assistant. Chat history is saved in `./resources/chat-history.mhtml` file.