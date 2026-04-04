import argparse
import json
import sys
from pathlib import Path
from typing import List

import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basit bir HF LLM ile metin üret ve MLflow'a kaydet."
    )
    parser.add_argument(
        "--prompt",
        default="Merhaba! Bana bu modelin neler yapabildiğini anlat.",
        help="Modelin kullanacağı başlangıç metni.",
    )
    parser.add_argument(
        "--model-name",
        default="sshleifer/tiny-gpt2",
        help="Hugging Face model adı (küçük bir model seçildi).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Üretilecek en fazla token sayısı.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=2,
        help="Kaç adet yanıt üretileceği.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling sıcaklığı (0-1 arası önerilir).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling değeri.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="İsteğe bağlı MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default="llm-demo",
        help="MLflow deney adı.",
    )
    parser.add_argument(
        "--run-name",
        default="tiny-gpt2-demo",
        help="MLflow run adı.",
    )
    return parser.parse_args()


def generate_responses(args: argparse.Namespace) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    outputs = generator(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [item["generated_text"] for item in outputs]


def main() -> None:
    args = parse_args()
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        responses = generate_responses(args)
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("max_new_tokens", args.max_new_tokens)
        mlflow.log_param("num_return_sequences", args.num_return_sequences)
        mlflow.log_param("temperature", args.temperature)
        mlflow.log_param("top_k", args.top_k)

        output_dir = Path("llm_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"llm_responses_{mlflow.active_run().info.run_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for text in responses:
                f.write(json.dumps({"prompt": args.prompt, "response": text}, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(output_path))

        sample_preview = responses[0] if responses else ""
        mlflow.log_text(sample_preview, artifact_file="llm_sample.txt")

        payload = json.dumps(
            {
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment": args.experiment_name,
                "run_id": mlflow.active_run().info.run_id,
                "artifact": str(output_path),
                "sample": sample_preview[:200],
            },
            ensure_ascii=False,
            indent=2,
        )
        # Jenkins Windows konsolu cp1254 ile yazarken encode hatası vermesin diye bytes yaz.
        sys.stdout.buffer.write(payload.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
