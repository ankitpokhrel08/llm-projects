import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
from pickle import load
import os

# Must be at the top
st.set_page_config(page_title="Image Captioning", layout="wide")

# ---------------------------------------------------------------------------
# Classes (must match notebook exactly for pickle compatibility)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split())
        self.word_to_idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        for idx, word in enumerate(sorted(vocab), start=4):
            self.word_to_idx[word] = idx
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

    @property
    def word_index(self):
        return self.word_to_idx

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_to_idx.get(word, self.word_to_idx['<unk>'])
                   for word in text.split()]
            sequences.append(seq)
        return sequences


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=256, hidden_dim=256, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()

        self.image_dropout = nn.Dropout(dropout)
        self.image_fc = nn.Linear(2048, embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.seq_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.decoder_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, vocab_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, input_seq):
        img_feat = self.image_dropout(image_features)
        img_feat = self.image_fc(img_feat)
        img_feat = self.relu(img_feat)

        seq_embed = self.embedding(input_seq)
        seq_embed = self.seq_dropout(seq_embed)
        lstm_out, (hidden, cell) = self.lstm(seq_embed)
        lstm_feat = hidden.squeeze(0)

        merged = img_feat + lstm_feat

        decoded = self.decoder_fc1(merged)
        decoded = self.relu(decoded)
        outputs = self.decoder_fc2(decoded)
        return outputs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- tokenizer ---
    tokenizer = load(open(os.path.join(script_dir, "tokenizer.p"), "rb"))
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 32

    # --- Xception feature extractor ---
    # MUST match training exactly:
    #   notebook cell 2:  model = timm.create_model('xception', pretrained=True)
    #   notebook cell 11: model = nn.Sequential(*list(model.children())[:-1])
    # Using num_classes=0 can produce DIFFERENT features depending on timm version.
    xception_full = timm.create_model('xception', pretrained=True)
    xception_model = nn.Sequential(*list(xception_full.children())[:-1])
    xception_model = xception_model.to(device)
    xception_model.eval()

    # --- Caption model ---
    caption_model = ImageCaptioningModel(vocab_size, max_length)
    checkpoint = torch.load(
        os.path.join(script_dir, 'models2', 'model_9.pth'),
        map_location=device,
    )
    caption_model.load_state_dict(checkpoint['model_state_dict'])
    caption_model = caption_model.to(device)
    caption_model.eval()

    return xception_model, caption_model, tokenizer, device, max_length, vocab_size


# ---------------------------------------------------------------------------
# Inference helpers  (copied verbatim from notebook)
# ---------------------------------------------------------------------------

def extract_features(image, model, device):
    """Extract features from a PIL image - same logic as notebook."""
    model.eval()

    image = image.resize((299, 299))
    image = np.array(image)
    image = image / 127.5 - 1.0                              # [-1, 1]
    image = torch.tensor(image).permute(2, 0, 1).float()     # (C, H, W)
    image = image.unsqueeze(0).to(device)                     # (1, C, H, W)

    with torch.no_grad():
        feature = model(image)

    return feature


def word_for_id(integer, tokenizer):
    """Convert integer back to word using tokenizer"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo, max_length, device):
    """Generate caption for image - EXACT copy from notebook."""
    model.eval()

    in_text = '<start>'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad sequence
        sequence = sequence + [0] * (max_length - len(sequence))
        sequence = sequence[:max_length]
        sequence = torch.tensor([sequence], dtype=torch.long).to(device)

        # Prepare photo tensor - exact same as notebook
        if isinstance(photo, dict):
            photo_tensor = list(photo.values())[0]
        else:
            photo_tensor = photo

        if len(photo_tensor.shape) == 3:
            photo_tensor = photo_tensor.unsqueeze(0)

        # Notebook always does torch.tensor() here (creates a fresh copy)
        photo_tensor = torch.tensor(photo_tensor, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(photo_tensor, sequence)
            pred = torch.argmax(pred, dim=1).item()

        word = word_for_id(pred, tokenizer)

        if word is None or word == '<end>':
            break

        in_text += ' ' + word

    caption = in_text.replace('<start>', '').strip()
    return caption


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.title("Image Captioning with Deep Learning")
    st.write("Upload an image and get an AI-generated caption!")

    # Load models (cached after first call)
    with st.spinner("Loading models..."):
        xception_model, caption_model, tokenizer, device, max_length, vocab_size = load_models()
    st.success("Models loaded successfully!")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses:\n"
        "- **Xception CNN** for image feature extraction\n"
        "- **LSTM** for caption generation\n"
        "- Trained on Flickr8k dataset"
    )
    st.sidebar.header("Model Info")
    st.sidebar.write(f"Vocabulary Size: {vocab_size}")
    st.sidebar.write(f"Max Caption Length: {max_length}")
    st.sidebar.write(f"Device: {device}")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to generate a caption",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Generated Caption")

            if st.button("Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    # Extract features
                    features = extract_features(image, xception_model, device)

                    # Generate caption
                    caption = generate_caption(
                        caption_model, tokenizer, features, max_length, device
                    )

                    # Store in session state so it persists across reruns
                    st.session_state['last_caption'] = caption

            # Show caption (persists after button click)
            if 'last_caption' in st.session_state:
                st.markdown(f"### *{st.session_state['last_caption']}*")


if __name__ == "__main__":
    main()
