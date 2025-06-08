def fec():
    import os
    import torch
    import pandas as pd
    from tqdm import tqdm
    from ultralytics import YOLO

    # === Load model ===
    model = YOLO("./my_yv10_5m/weights/best.pt")
    model.eval()

    # === Hook to capture features ===
    features_dict = {}

    def hook_fn(module, input, output):
        pooled = torch.mean(output[0], dim=(1, 2))  # Global average pooling over H, W
        features_dict['feat'] = pooled.detach().cpu().numpy()

    # Register hook to internal feature map layer (adjust index if needed)
    hook = model.model.model[10].register_forward_hook(hook_fn)

    # === Extract feature from a single image ===
    def extract_feature_from_image(img_path, save_csv_path):
        filename = os.path.basename(img_path)

        try:
            _ = model(img_path)  # Inference triggers the hook
            feat = features_dict.get('feat')
            #print('len',len(feat))
            if feat is not None:
                feat = feat.flatten()
                feat_row = [filename] + feat.tolist()
                columns = ['filename'] + [f'feat_{i}' for i in range(len(feat))]
                df = pd.DataFrame([feat_row], columns=columns)
                df.to_csv(save_csv_path, index=False)
                #print(f"✅ Feature saved to {save_csv_path}")
            else:
                print('')
                #print("⚠️ Feature not extracted. Check layer index.")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # === Example Usage ===
    extract_feature_from_image(
        img_path="./images/input.png",
        save_csv_path="./input_feature_unlabeled.csv"
    )
