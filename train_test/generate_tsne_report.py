import argparse
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def build_color_mapping(cluster_ids):

    # Plotly qualitative palette (extended)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"
    ]

    unique_clusters = []
    seen = set()
    for cid in cluster_ids:
        if cid not in seen:
            seen.add(cid)
            unique_clusters.append(cid)

    color_map = {}
    for idx, cid in enumerate(unique_clusters):
        if idx < len(palette):
            color_map[cid] = palette[idx]
        else:
            # Fallback: deterministic extra colors via HSV rotation
            # Keep hue cycling distinct enough
            import colorsys
            hue = (idx - len(palette)) * 0.123 % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
            color_map[cid] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return color_map


def generate_report(tsne_json_path, test_set_csv, previews_dir, output_html):

    with open(tsne_json_path, "r") as f:
        data = json.load(f)

    x = data["x"]
    y = data["y"]
    track_ids = data["track_ids"]
    cluster_ids = data["cluster_ids"]

    # Build metadata mapping from CSV
    df = pd.read_csv(test_set_csv)
    meta = df.set_index("trackID")[ ["name", "artist"] ]

    names = []
    artists = []
    for tid in track_ids:
        if tid in meta.index:
            row = meta.loc[tid]
            names.append(str(row["name"]))
            artists.append(str(row["artist"]))
        else:
            names.append(str(tid))
            artists.append("")

    # Colors per cluster (no legend)
    color_map = build_color_mapping(cluster_ids)
    point_colors = [color_map[cid] for cid in cluster_ids]

    # Compute preview path relative to report location
    output_dir = os.path.dirname(os.path.abspath(output_html))
    previews_rel = os.path.relpath(previews_dir, start=output_dir)
    preview_paths = [os.path.join(previews_rel, f"{tid}.wav") for tid in track_ids]

    # Customdata: [track_id, preview_rel_path, name, artist]
    customdata = [
        [tid, ppath, n, a]
        for tid, ppath, n, a in zip(track_ids, preview_paths, names, artists)
    ]

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=point_colors,
                    size=6,
                    opacity=0.85
                ),
                customdata=customdata,
                hovertemplate="%{customdata[2]} — %{customdata[3]}<extra></extra>",
                showlegend=False,
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="t-SNE colored by ground-truth clusterID", x=0.5, y=0.95, xanchor="center"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    # Build HTML with audio element and hover handlers
    fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="tsne_plot")

    extra_html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>t-SNE Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; }}
    #container {{ max-width: 1100px; margin: 0 auto; }}
    #player-wrap {{ position: sticky; top: 8px; background: #fafafa; padding: 8px; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 10px; }}
    #hover-label {{ font-size: 14px; margin-left: 8px; color: #333; }}
  </style>
  <script>
  document.addEventListener('DOMContentLoaded', function() {{
    var plot = document.getElementById('tsne_plot');
    var audio = document.getElementById('hover-audio');
    var label = document.getElementById('hover-label');

    if (plot && plot.on) {{
      plot.on('plotly_hover', function(evt) {{
        if (!evt || !evt.points || evt.points.length === 0) return;
        var p = evt.points[0];
        var cd = p.customdata || [];
        var src = cd[1];
        var name = cd[2] || '';
        var artist = cd[3] || '';
        if (src) {{
          if (audio.getAttribute('src') !== src) {{
            audio.setAttribute('src', src);
          }}
          try {{ audio.currentTime = 0; }} catch (e) {{}}
          var playPromise = audio.play();
          if (playPromise && playPromise.catch) {{ playPromise.catch(function(){{}}); }}
        }}
        label.textContent = name + (artist ? ' — ' + artist : '');
      }});
      plot.on('plotly_unhover', function() {{
        try {{ audio.pause(); }} catch (e) {{}}
      }});
    }}
  }});
  </script>
</head>
<body>
  <div id=\"container\">
    <div id=\"player-wrap\"> 
      <audio id=\"hover-audio\" controls preload=\"auto\"></audio>
      <span id=\"hover-label\"></span>
    </div>
    {fig_html}
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(extra_html)

    print(f"Wrote HTML report to: {output_html}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tsne_json", required=True, help="Path to t-SNE JSON with x,y,track_ids,cluster_ids")
    parser.add_argument("--test_set", required=True, help="CSV with columns trackID,name,artist")
    parser.add_argument("--previews_dir", required=True, help="Directory containing {trackID}.wav previews")
    parser.add_argument("--output_html", required=True, help="Where to write the HTML report")
    args = parser.parse_args()

    generate_report(
        tsne_json_path=args.tsne_json,
        test_set_csv=args.test_set,
        previews_dir=args.previews_dir,
        output_html=args.output_html,
    )


if __name__ == "__main__":
    main()


