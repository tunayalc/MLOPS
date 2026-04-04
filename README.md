# MLOPS

Model eğitimi, izlenebilirlik, fairness ve MLSecOps kontrollerini birleştiren uçtan uca MLOps çalışması

## Genel Bakış

`MLOPS`, yalnızca model eğitimi yapan bir repo değil; model yaşam döngüsünü daha disiplinli hale getirmek için deney takibi, veri soy kütüğü, fairness raporlaması, güvenlik kontrolleri, CI akışı ve yönetişim çıktıları üreten daha geniş bir MLOps / MLSecOps çalışması.

Bu repo, `YZM315 Yapay Zeka İçin Yazılım Mühendisliği` dersi kapsamında öğrendiklerimizi uygulamaya döktüğümüz çalışma alanlarından biriydi.

Repo içindeki örnek senaryo, öğrenci performansı verisi üzerinde regresyon modeli eğitmek üzerine kurulu. Ancak asıl odak veri setinin kendisinden çok, modelin operasyonel yönetişimini de kapsayan süreç tasarımı.

## Kapsanan Başlıklar

| Alan | Repo İçindeki Karşılığı |
| --- | --- |
| Model eğitimi | `train_pipeline.py` |
| Yeniden eğitim | `retrain_pipeline.py` |
| Deney ve veri kaydı | `experiment_store.py`, MLflow, SQLite |
| Veri alma ve preprocessing | `pipeline_utils.py` |
| Regresyon çıktıları ve artefact üretimi | `regression_reporting.py` |
| Fairness analizi | `fairness_reporting.py` |
| MLSecOps guardrail'leri | `mlsecops_guardrails.py` |
| Giskard tabanlı model taraması | `giskard_scan.py` |
| LLM demo hattı | `llm_pipeline.py` |
| Credo AI tarzı manifest üretimi | `credo_manifest.py` |
| Pipeline orkestrasyonu | `dvc.yaml` |
| CI/CD akışı | `Jenkinsfile` |

## Mimari Akış

Repo içindeki operasyonel düşünce şu sırayla ilerliyor:

1. veri seti indirilir veya cache'den alınır
2. preprocessing + model pipeline kurulur
3. eğitim ve değerlendirme yapılır
4. train/test split'leri ve metadata SQLite'a yazılır
5. MLflow üzerinde deney kaydı açılır
6. fairness raporu üretilir
7. MLSecOps raporu oluşturulur
8. Giskard ve Garak gibi ek tarama katmanları devreye alınır
9. SBOM ve yönetişim manifest'i üretilir

Bu yönüyle repo, klasik “train.py çalıştır ve sonucu al” seviyesinden daha ileri bir yaşam döngüsü örneği sunuyor.

## Temel Bileşenler

### `pipeline_utils.py`

UCI Student Performance veri setini indiren, cache'leyen ve eğitim pipeline'ını kuran yardımcı katman. Sayısal ve kategorik alanları ayırıp `ColumnTransformer` ve `RandomForestRegressor` temelli pipeline üretimi burada yapılıyor.

### `train_pipeline.py`

Ana eğitim akışı. Bu dosya:

- veri setini yükler
- train/test split yapar
- modeli eğitir
- temel regresyon metriklerini hesaplar
- robustness ve generalization kontrollerini çalıştırır
- fairness raporu üretir
- MLflow'a log atar
- SQLite deney deposuna metadata ve veri snapshot'larını yazar

### `retrain_pipeline.py`

Önceki bir deneyin metadata'sını SQLite üzerinden geri okuyup aynı mantıkla yeniden eğitim yapan akış. Bu tasarım, deneyin yalnızca sonucu değil, tekrar üretilebilirliğini de önemseyen bir yaklaşım taşıyor.

### `experiment_store.py`

Deney metadata'sını ve train/test veri snapshot'larını SQLite içinde tutan depo katmanı. Bu dosya repo için kritik; çünkü sadece model sonucu değil, kullanılan verinin de kayıt altına alınmasını sağlıyor.

### `regression_reporting.py`

Tahmin / gerçek karşılaştırması, residual analizi ve feature importance görselleri gibi regresyon odaklı artefact'lar üreten raporlama katmanı.

### `fairness_reporting.py`

`Fairlearn` kullanarak grup bazlı regresyon metrikleri üretiyor. Varsayılan duyarlı özellik olarak `sex` kolonu üzerinden grup kırılımı ve disparity hesapları yapılıyor.

### `mlsecops_guardrails.py`

Repo'nun en ayırt edici dosyalarından biri. Bu katman:

- veri lineage kaydı
- veri hash doğrulaması
- manifest kontrolü
- model bütünlüğü için hash üretimi
- noise robustness testi
- generalization gap analizi
- OWASP ML Top 10 ve MITRE ATLAS referanslı güvenlik raporu

gibi kontrolleri bir araya getiriyor.

### `giskard_scan.py`

Son eğitilmiş modeli MLflow üzerinden geri yükleyip Giskard ile tarayan ek analiz katmanı.

### `llm_pipeline.py`

Repo'nun opsiyonel LLM tarafı. Hugging Face modeliyle küçük bir text-generation akışı kuruyor ve çıktıları yine MLflow üzerinden logluyor. Böylece repo yalnızca klasik ML değil, LLM operasyonlarına da küçük bir geçiş alanı sunuyor.

### `credo_manifest.py`

MLSecOps raporu, fairness çıktısı, Giskard taraması ve SBOM gibi üretilmiş artefact'ları tek manifest içinde toplayan yönetişim odaklı yardımcı dosya.

### `dvc.yaml`

DVC pipeline tanımı. Eğitim akışının bağımlılıklarını ve çıktılarını tanımlayarak deney tekrar üretilebilirliğini artıran orkestrasyon katmanı.

### `Jenkinsfile`

CI/CD tarafında şu adımları koşan otomasyon akışı bulunuyor:

- Python ortamı hazırlama
- statik kontrol / compile doğrulaması
- model eğitimi
- smoke retrain
- MLSecOps raporu kontrolü
- fairness raporu kontrolü
- Giskard taraması
- LLM demo akışı
- Garak taraması
- CycloneDX SBOM üretimi
- Credo AI manifest üretimi
- DVC snapshot güncellemesi

Bu, repo'nun yalnızca lokal script seti değil, pipeline odaklı düşünüldüğünü açıkça gösteriyor.

## Repo Yapısı

```text
MLOPS/
|-- train_pipeline.py
|-- retrain_pipeline.py
|-- fairness_reporting.py
|-- regression_reporting.py
|-- mlsecops_guardrails.py
|-- giskard_scan.py
|-- llm_pipeline.py
|-- pipeline_utils.py
|-- experiment_store.py
|-- credo_manifest.py
|-- dvc.yaml
|-- dvc.lock
|-- Jenkinsfile
|-- requirements.txt
|-- requirements-llm.txt
|-- garak_config.yaml
|-- mlsecops_manifest.json
`-- README.md
```

## Kullanılan Teknolojiler

### Core ML / MLOps

- Python
- scikit-learn
- pandas
- NumPy
- MLflow
- DVC
- SQLite
- matplotlib

### Responsible AI / Security

- Fairlearn
- Giskard
- CycloneDX
- Garak

### Opsiyonel LLM Katmanı

- Hugging Face Transformers
- PyTorch
