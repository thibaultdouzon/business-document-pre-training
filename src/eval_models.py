import argparse
import json

from pathlib import Path

from src import train, prediction, dataset_util, icdar_sroie, documents, label_config, utils


def eval_model(test_dataset, dp_postprocess, i, save_dir, model_name="layoutlm"):
    model = train.get_model(saved_model=f"saved_models/{save_dir}_{i}", model_name=model_name)

    *results, preds = prediction.evaluate(model, test_dataset, use_dp=dp_postprocess)
    
    with open(Path("saved_models") / f"{save_dir}_{i}" / "results.json", "w") as fp:
        json.dump(results, fp, cls=utils.DataclassJSONEncoder)
    with open(Path("saved_models") / f"{save_dir}_{i}" / "predictions.json", "w") as fp:
        json.dump([pred.export() for pred in preds], fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",)
    parser.add_argument("--num_runs", type=int)
    parser.add_argument("--dp_postprocess", type=bool, default=True)

    args = parser.parse_args()

    if label_config.conf["LABEL"]["Name"] ==  "ICDAR":
        test_dataset = dataset_util.DocumentDataset(icdar_sroie.get_docs_from_disk(Path("data/icdar/test")))
    elif label_config.conf["LABEL"]["Name"] == "BDCPO":
        test_dataset = dataset_util.DocumentDataset(documents.load_from_layoutlm_style_dataset(Path("/data/bdcpo"), "test"))
    else:
        raise Exception("Not available")

    for i in range(args.num_runs):
        eval_model(test_dataset, args.dp_postprocess, i, args.save_dir)
   


if __name__ == '__main__':
    main()