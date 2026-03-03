
1. **Clone the repository**

   ```bash

   ```

2. **Install dependencies**

   ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tokenizers
pip install faiss-cpu
pip install faiss-gpu
pip install numpy scikit-learn tqdm
   ```

3. **Run the development server**

   ```bash
   npm run dev
   ```

## 📁 Project Structure

```
Project/
  rvl-cdip
  rvl-cdip-o
  rvl-cdip-o-text
  QS-OCR-Large
  test.txt
  train.txt
  val.txt
  instruction.txt
  src/
    config.py
    data.py
    sampler.py
    model.py
    loss.py
    knn_ood.py
    metrics.py
    train.py
    evaluate.py
```
