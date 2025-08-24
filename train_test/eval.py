from tqdm import tqdm
import argparse
import pandas as pd
from utils import download_and_preprocess_previews
import torch
import json
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.cluster import contingency_matrix
#from matplotlib.colors import get_cmap
from generate_tsne_report import generate_report

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {device}')

def purity_scores(y_true, y_pred):
    C = contingency_matrix(y_true, y_pred)
    purity_pred = np.sum(np.max(C, axis=0)) / np.sum(C)   # purity by predicted clusters
    purity_true = np.sum(np.max(C, axis=1)) / np.sum(C)   # inverse purity by true clusters
    return purity_pred, purity_true

def hungarian_matched_accuracy(y_true, y_pred):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return None
    C = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(C.max() - C)
    return C[row_ind, col_ind].sum() / C.sum()

def clustering_report(y_true, y_pred, X=None):
    report = {}
    # Label-invariant external metrics
    report["ari"] = adjusted_rand_score(y_true, y_pred)
    report["nmi"] = normalized_mutual_info_score(y_true, y_pred)
    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    report["homogeneity"] = h
    report["completeness"] = c
    report["v_measure"] = v
    report["fmi"] = fowlkes_mallows_score(y_true, y_pred)

    # Purity metrics
    purity, inv_purity = purity_scores(y_true, y_pred)
    report["purity"] = purity
    report["inverse_purity"] = inv_purity

    # Best label mapping accuracy (intuitive “how many in right bucket”)
    report["hungarian_matched_accuracy"] = hungarian_matched_accuracy(y_true, y_pred)

    # Intrinsic metrics (ignore labels)
    if X is not None and len(np.unique(y_pred)) > 1:
        # cosine often matches your k-means distance choice
        report["silhouette_cosine"] = silhouette_score(X, y_pred, metric="cosine")
        report["calinski_harabasz"] = calinski_harabasz_score(X, y_pred)
        report["davies_bouldin"] = davies_bouldin_score(X, y_pred)
    return report

def load_model(model_name): 
    if model_name == "mert_base":
        from models.mert.mert import MERT
        return MERT('m-a-p/MERT-v1-330M')
    elif model_name == 'clmr_base':
        from models.clmr.sample_cnn import SampleCNN
        from models.clmr.utils import load_encoder_checkpoint
        strides = [3, 3, 3, 3, 3, 3, 3, 3, 3]
        # We want encoder features (512-d). Bypass the fc layer.
        out_dim = 512
        model_path = 'models/clmr/checkpoints/clmr_checkpoint_10000.pt'
        print(f'Loading CLMR base model from {model_path}')
        sample_cnn = SampleCNN(strides, False, out_dim)
        state_dict = load_encoder_checkpoint(model_path, out_dim)
        sample_cnn.load_state_dict(state_dict)
        # Replace classification head with identity to expose 512-d embeddings
        import torch.nn as nn
        sample_cnn.fc = nn.Identity()
        return sample_cnn
    else:
        return None

def main():

    #get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, default="../data/test_tracks.csv")
    parser.add_argument("--previews_dir", type=str, default="previews")
    parser.add_argument("--model_name", type=str, default="mert_base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--embeddings_path", type=str, default=None)
    args = parser.parse_args()

    #load test set
    df = pd.read_csv(args.test_set)
    df = df[df['previewURL'] != 'None']
    print(df.head())
   
    #download previews
    print('df cluster ids: ', df['clusterID'].tolist())
    if args.model_name == 'clmr_base':
        from models.clmr.utils import clmr_load_wavs
        load_wavs = clmr_load_wavs
        sample_rate = 22050
        embedding_dim = 512
    elif args.model_name == 'mert_base':
        from models.mert.utils import mert_load_wavs
        load_wavs = mert_load_wavs
        sample_rate = 24000
        embedding_dim = 1024
    wav_paths, cluster_ids, track_ids = download_and_preprocess_previews(df['previewURL'].tolist(), df['trackID'].tolist(),  df['clusterID'].tolist(), args.previews_dir, sample_rate=sample_rate)
    

    print(args.embeddings_path)

    if args.embeddings_path is None:
        #load model
        model = load_model(args.model_name)
        model = model.to(device)
        model.eval()

        #generate embeddings in batches (512-d from encoder)
        all_embeddings = torch.zeros(len(wav_paths), embedding_dim)
        for i in tqdm(range(0, len(wav_paths), args.batch_size)):      
            batch = load_wavs(wav_paths[i:min(i+args.batch_size, len(wav_paths))])
            if args.model_name == 'clmr_base':
                batch = batch.to(device)
            with torch.no_grad():
                outputs = model(batch)
                embeddings = outputs.detach().cpu()
                all_embeddings[i:min(i+args.batch_size, len(wav_paths))] = embeddings

        # Save embeddings tensor
        torch.save(all_embeddings, f'eval_results/embeddings_{args.model_name}.pt')
        print(f'all_embeddings.shape: {all_embeddings.shape}')
    else:
        all_embeddings=torch.load(args.embeddings_path)

    #run tSNE
    data = all_embeddings.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
    tsne_results = tsne.fit_transform(data)
    
    tsne_results = np.asarray(tsne_results)
    cluster_ids = np.asarray(cluster_ids)
    print(len(tsne_results))
    print(len(cluster_ids))

    unique_ids = np.unique(cluster_ids)
    cmap = plt.get_cmap('tab20', len(unique_ids))  # discrete colors

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cid in enumerate(unique_ids):
        mask = (cluster_ids == cid)
        ax.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            s=12,
            color=cmap(idx),
            label=str(cid),
            alpha=0.85,
            edgecolors='none'
        )

    # No grid lines or axis titles/ticks
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend to the side (large)
    ax.legend(
        title="cluster_id",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        ncol=1,
        fontsize=12,
        title_fontsize=13,
        markerscale=1.8
    )

    fig.tight_layout(rect=[0, 0, 0.8, 1])  # room for legend on right
    os.makedirs("eval_results", exist_ok=True)
    plt.show()
    fig.savefig(f"eval_results/{args.model_name}_tsne.png", dpi=200, bbox_inches="tight")

    # Save t-SNE data for HTML report and generate interactive report
    tsne_json_path = f"eval_results/{args.model_name}_tsne.json"
    tsne_payload = {
        "x": tsne_results[:, 0].tolist(),
        "y": tsne_results[:, 1].tolist(),
        "track_ids": list(track_ids),
        "cluster_ids": list(cluster_ids)
    }
    with open(tsne_json_path, "w") as fjson:
        json.dump(tsne_payload, fjson)

    output_html = f"eval_results/{args.model_name}_tsne_report.html"
    generate_report(
        tsne_json_path=tsne_json_path,
        test_set_csv=args.test_set,
        previews_dir=args.previews_dir,
        output_html=output_html,
    )



    #evaluate via k-means
    kmeans_cluster_ids, cluster_centers = kmeans(
        X=all_embeddings,
        num_clusters=args.n_clusters,
        distance='cosine',
        device=device
    )
    print(kmeans_cluster_ids)

    metrics = clustering_report(y_true=cluster_ids, y_pred=kmeans_cluster_ids, X=all_embeddings)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(f'eval_results/{args.model_name}_report.json', 'w') as f:
        json.dump(metrics, f)




if __name__ == "__main__":
    main()
