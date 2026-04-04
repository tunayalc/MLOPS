pipeline {
    agent any

    environment {
        PYTHONPATH = "${WORKSPACE}"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            python3 -m venv "$VENV"
                            . "$VENV/bin/activate"
                            pip install --upgrade pip
                            if [ -f requirements.txt ]; then
                                pip install -r requirements.txt
                            else
                                pip install pandas scikit-learn mlflow matplotlib
                            fi
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            if (-not (Test-Path $venv)) { python -m venv $venv }
                            $py = Join-Path $venv "Scripts\\python.exe"
                            & $py -m pip install --upgrade pip
                            if (Test-Path "requirements.txt") {
                                & $py -m pip install -r requirements.txt
                            } else {
                                & $py -m pip install pandas scikit-learn mlflow matplotlib
                            }
                            if (Test-Path "requirements-llm.txt") {
                                & $py -m pip install -r requirements-llm.txt
                            }
                        '''
                    }
                }
            }
        }

        stage('Static Checks') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            . "$VENV/bin/activate"
                            python -m compileall train_pipeline.py retrain_pipeline.py regression_reporting.py experiment_store.py pipeline_utils.py mlsecops_guardrails.py
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            & $py -m compileall train_pipeline.py retrain_pipeline.py regression_reporting.py experiment_store.py pipeline_utils.py mlsecops_guardrails.py
                        '''
                    }
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            DB_PATH="${WORKSPACE}/experiments.db"
                            MLFLOW_DB="${WORKSPACE}/experiments.mlflow.db"
                            . "$VENV/bin/activate"
                            python train_pipeline.py --mlflow-tracking-uri "sqlite:///${MLFLOW_DB}" --db-path "${DB_PATH}"
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            $db = Join-Path $env:WORKSPACE "experiments.db"
                            $mlflowDb = Join-Path $env:WORKSPACE "experiments.mlflow.db"
                            $mlflowUri = "sqlite:///" + ($mlflowDb.Replace('\\', '/'))
                            & $py train_pipeline.py --mlflow-tracking-uri $mlflowUri --db-path $db
                        '''
                    }
                }
            }
        }

        stage('Retrain (Smoke)') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            DB_PATH="${WORKSPACE}/experiments.db"
                            MLFLOW_DB="${WORKSPACE}/experiments.mlflow.db"
                            . "$VENV/bin/activate"
                            python retrain_pipeline.py --mlflow-tracking-uri "sqlite:///${MLFLOW_DB}" --db-path "${DB_PATH}" --reuse-mlflow-uri
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            $db = Join-Path $env:WORKSPACE "experiments.db"
                            $mlflowDb = Join-Path $env:WORKSPACE "experiments.mlflow.db"
                            $mlflowUri = "sqlite:///" + ($mlflowDb.Replace('\\', '/'))
                            & $py retrain_pipeline.py --mlflow-tracking-uri $mlflowUri --db-path $db --reuse-mlflow-uri
                        '''
                    }
                }
            }
        }

        stage('MLSecOps Audit') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            . "$VENV/bin/activate"
                            REPORT=$(ls -1t mlsecops_reports/*_mlsecops_report.json 2>/dev/null | head -n 1)
                            if [ -z "$REPORT" ]; then
                                echo "No MLSecOps report found under mlsecops_reports/"
                                exit 1
                            fi
                            echo "MLSecOps report: $REPORT"
                            head -n 40 "$REPORT"
                        '''
                    } else {
                        powershell '''
                            $report = Get-ChildItem "mlsecops_reports\\*_mlsecops_report.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
                            if (-not $report) {
                                Write-Error "No MLSecOps report found under mlsecops_reports\\"
                                exit 1
                            }
                            Write-Host ("MLSecOps report: {0}" -f $report.FullName)
                            Get-Content $report.FullName -TotalCount 40 | ForEach-Object { Write-Host $_ }
                        '''
                    }
                }
            }
        }

        stage('Fairness Audit (Fairlearn)') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            REPORT=$(ls -1t fairness_reports/*_fairlearn_fairness_report.json 2>/dev/null | head -n 1)
                            if [ -z "$REPORT" ]; then
                                echo "No Fairlearn fairness report found under fairness_reports/"
                                exit 1
                            fi
                            echo "Fairlearn fairness report: $REPORT"
                            head -n 60 "$REPORT"
                        '''
                    } else {
                        powershell '''
                            $report = Get-ChildItem "fairness_reports\\*_fairlearn_fairness_report.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
                            if (-not $report) {
                                Write-Error "No Fairlearn fairness report found under fairness_reports\\"
                                exit 1
                            }
                            Write-Host ("Fairlearn fairness report: {0}" -f $report.FullName)
                            Get-Content $report.FullName -TotalCount 60 | ForEach-Object { Write-Host $_ }
                        '''
                    }
                }
            }
        }

        stage('Giskard Scan (Tabular)') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            DB_PATH="${WORKSPACE}/experiments.db"
                            . "$VENV/bin/activate"
                            mkdir -p giskard_reports
                            python giskard_scan.py --db-path "${DB_PATH}" --output-dir "giskard_reports"
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            $db = Join-Path $env:WORKSPACE "experiments.db"
                            $reportDir = Join-Path $env:WORKSPACE "giskard_reports"
                            if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Path $reportDir | Out-Null }
                            & $py giskard_scan.py --db-path $db --output-dir "giskard_reports"
                        '''
                    }
                }
            }
        }

        stage('LLM Demo') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            DB_PATH="${WORKSPACE}/experiments.db"
                            MLFLOW_DB="${WORKSPACE}/experiments.mlflow.db"
                            . "$VENV/bin/activate"
                            python llm_pipeline.py --model-name sshleifer/tiny-gpt2 --max-new-tokens 64 --num-return-sequences 2 --mlflow-tracking-uri "sqlite:///${MLFLOW_DB}"
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            $mlflowDb = Join-Path $env:WORKSPACE "experiments.mlflow.db"
                            $mlflowUri = "sqlite:///" + ($mlflowDb.Replace('\\', '/'))
                            & $py llm_pipeline.py --model-name sshleifer/tiny-gpt2 --max-new-tokens 64 --num-return-sequences 2 --mlflow-tracking-uri $mlflowUri
                        '''
                    }
                }
            }
        }

        stage('Garak Scan') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            . "$VENV/bin/activate"
                            mkdir -p garak_reports
                            python -X utf8 -m garak \
                              --model_type huggingface \
                              --model_name sshleifer/tiny-gpt2 \
                              --probes divergence.Repeat \
                              --generations 1
                            # garak bazı sürümlerde raporu $HOME/.local altına yazar; workspace'e kopyalayalım
                            if [ -d "$HOME/.local/share/garak/garak_runs/garak_reports" ]; then
                              cp -r "$HOME/.local/share/garak/garak_runs/garak_reports/"* garak_reports/ 2>/dev/null || true
                            fi
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            $reportDir = Join-Path $env:WORKSPACE "garak_reports"
                            if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Path $reportDir | Out-Null }
                            & $py -X utf8 -m garak --model_type huggingface --model_name sshleifer/tiny-gpt2 --probes divergence.Repeat --generations 1
                            $homeReports = Join-Path $env:USERPROFILE ".local\\share\\garak\\garak_runs\\garak_reports"
                            if (Test-Path $homeReports) {
                                Copy-Item (Join-Path $homeReports "*") $reportDir -Recurse -Force -ErrorAction SilentlyContinue
                            }
                        '''
                    }
                }
            }
        }

        stage('SBOM (CycloneDX)') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            . "$VENV/bin/activate"
                            SBOM_PATH="${WORKSPACE}/cyclonedx_sbom.json"
                            python -m cyclonedx_py environment "$VENV/bin/python" --of JSON -o "$SBOM_PATH"
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            $sbom = Join-Path $env:WORKSPACE "cyclonedx_sbom.json"
                            & $py -m cyclonedx_py environment $py --of JSON -o $sbom
                        '''
                    }
                }
            }
        }

        stage('Credo AI Manifest') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            . "$VENV/bin/activate"
                            python credo_manifest.py
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            & $py credo_manifest.py
                        '''
                    }
                }
            }
        }

        stage('DVC Snapshot') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            VENV="${WORKSPACE}/.venv"
                            . "$VENV/bin/activate"
                            if [ ! -d ".dvc" ]; then
                                dvc init -q
                            fi
                            dvc repro
                            git config user.name "Jenkins CI"
                            git config user.email "jenkins@example.com"
                            if [ -f dvc.lock ] && ! git diff --quiet -- dvc.lock; then
                                git add dvc.lock
                                git commit -m "ci: update dvc lock [skip ci]" || true
                            fi
                        '''
                    } else {
                        powershell '''
                            $venv = Join-Path $env:WORKSPACE ".venv"
                            $py = Join-Path $venv "Scripts\\python.exe"
                            if (-not (Test-Path ".dvc")) {
                                & $py -m dvc init -q
                            }
                            & $py -m dvc repro
                            git config user.name "Jenkins CI"
                            git config user.email "jenkins@example.com"
                            if (Test-Path "dvc.lock") {
                                git diff --quiet -- dvc.lock
                                if ($LASTEXITCODE -ne 0) {
                                    git add dvc.lock
                                    git commit -m "ci: update dvc lock [skip ci]" | Out-Null
                                }
                            }
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'mlruns/**', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: '*.db', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'mlsecops_reports/*.json', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'llm_outputs/*.jsonl', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'garak_reports/**', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'fairness_reports/*.json', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'giskard_reports/**', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'cyclonedx_sbom.json', fingerprint: true, allowEmptyArchive: true
            archiveArtifacts artifacts: 'credo_ai_manifest.json', fingerprint: true, allowEmptyArchive: true
        }
        success {
            echo 'Pipeline completed successfully.'
        }
        failure {
            echo 'Pipeline failed. Check console output for details.'
        }
    }
}
