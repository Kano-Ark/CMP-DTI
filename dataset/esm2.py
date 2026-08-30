import torch
import esm
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Mamba 模型名称
#MODEL_NAME = "Rostlab/prot_bert"
#MODEL_NAME = "ProtTrans/prot_t5_xl_bfd"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型（这里使用 ESM-2 650M 1280维）
def load_esm2_model():
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

# 提取序列的平均池化嵌入
def get_embedding(model, batch_converter, seq_id, sequence):
    data = [(seq_id, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # 去掉 [CLS] 和 [EOS]，只取中间的氨基酸表示，做平均池化
    representation = token_representations[0, 1:len(sequence)+1].mean(0).cpu()
    
    # ✅ L2归一化（每行模长为1）
    # representation = representation / representation.norm(p=2)
    
    # ✅ Min-Max归一化（每个特征值在0-1之间）
    representation = token_representations[0, 1:len(sequence)+1].mean(0).cpu()

    return representation

def main():
    input_file = "dataset/kiba/protein_seq.csv"
    output_file = "dataset/kiba/normalized_seq_esm.csv"

    df = pd.read_csv(input_file, header=None)
    assert df.shape[1] >= 2, "蛋白质CSV必须包含至少两列（ID 和序列）"

    model, batch_converter = load_esm2_model()
    print("模型加载完成，开始处理序列...")

    embeddings = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row[0])
        seq = str(row[1]).strip().upper()
        if len(seq) < 10:
            print(f"警告：序列过短，跳过 ID = {pid}")
            continue
        emb = get_embedding(model, batch_converter, pid, seq)
        emb_list = emb.tolist()
        embeddings.append([pid] + emb_list)
        
    # Min-Max 归一化到 [0, 1]
    embeddings_ids = [row[0] for row in embeddings]
    embeddings_vectors = [row[1:] for row in embeddings]
    scaler = MinMaxScaler()
    embeddings_vectors = scaler.fit_transform(embeddings_vectors)
    # 重新组合ID和归一化后的向量
    embeddings_normalized = [[pid] + vec.tolist() for pid, vec in zip(embeddings_ids, embeddings_vectors)]
    
    # 保存为CSV
    out_df = pd.DataFrame(embeddings_normalized)
    out_df.to_csv(output_file, index=False, header=False)
    print(f"特征已保存至: {output_file}")

if __name__ == "__main__":
    main()