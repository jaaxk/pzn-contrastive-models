import argparse
import pandas as pd
from utils import download_and_preprocess_previews
from models.mert import MERT
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans



def load_model(model_name):
    if model_name == "mert_base":
        return MERT('m-a-p/MERT-v1-330M')
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
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE: {device}')

    #download previews
    wav_paths = download_and_preprocess_previews(df['previewURL'].tolist(), df['trackID'].tolist(), args.previews_dir)
    print(wav_paths)

    print(args.embeddings_path)

    if args.embeddings_path is None:
        #load model
        model = load_model(args.model_name)

        #generate embeddings in batches
        all_embeddings = torch.zeros(len(wav_paths), 1024)

        for i in range(0, len(wav_paths), args.batch_size):
            batch = wav_paths[i:min(i+args.batch_size, len(wav_paths))]
            with torch.no_grad():
                embeddings = model(batch)
                all_embeddings[i:min(i+args.batch_size, len(wav_paths))] = embeddings

        # Save embeddings tensor
        torch.save(all_embeddings, f'embeddings_{args.model_name}.pt')
        print(all_embeddings.shape)
    else:
        all_embeddings=torch.load(args.embeddings_path)

    #run tSNE
    data = all_embeddings.cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
    tsne_results = tsne.fit_transform(data)
    #plot
    plt.figure(figsize=(8,6))
    plt.scatter(tsne_results[:,0], tsne_results[:,1])
    plt.title('t-SNE Plot of PyTorch Tensor')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.show()

    #evaluate via k-means
    cluster_ids, cluster_centers = kmeans(
        X=all_embeddings,
        num_clusters=args.n_clusters,
        distance='cosine',
        device=device
    )
    print(cluster_ids)




if __name__ == "__main__":
    main()