# Cloud Run Jobs for the 3-stage ETL + analysis pipeline (ingest -> transform -> analyse).
#
# Explicitly out of scope: no 4th "model"/"training" job exists here or anywhere in this
# file. Modeling has not started in this project; nothing in this pipeline may auto-fire it.
#
# All 3 jobs run as the existing oam-pipeline-sa (google_service_account.oam_pipeline_sa in
# runtime_identity.tf) -- no new service account is created here.

locals {
  artifact_registry = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.oam_containers.repository_id}"
  # Per-job tags (Artifact Registry repo has immutable_tags=true, so a fixed build gets a
  # new tag rather than overwriting v1). Bumped based on real end-to-end validation:
  # oam-analyse (v2) was missing a transitive pyarrow dependency
  # (odi/contracts.py -> pipelines/silver/contracts.py). oam-transform went through three
  # fixes: v2 added `git` + a resolvable commit for build_statsbomb_silver.py's provenance
  # stamping (`_git_sha()`); v3 fixed the entrypoint's own DATASET-routing patch (see
  # scripts/run_oam_transform.py), which had been overwriting the *shared*
  # audit_cxg_e13_f1_f15.DATASET constant instead of only materialize_gold_cxg's own copied
  # binding, breaking that module's oam_core reads; v4 fixes a SyntaxError in
  # materialize_cxg_feature_family_tables.py (backslash inside an f-string expression --
  # invalid on Python 3.11, the container's runtime) that deterministically crashed the
  # feature-family-materialization step every run.
  pipeline_image_tags = {
    oam-ingest    = "v1"
    oam-transform = "v4"
    oam-analyse   = "v2"
  }
}

resource "google_cloud_run_v2_job" "oam_ingest" {
  name                = "oam-ingest"
  project             = var.project_id
  location            = var.region
  deletion_protection = false
  labels              = local.common_labels

  template {
    template {
      service_account = google_service_account.oam_pipeline_sa.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = "${local.artifact_registry}/oam-ingest:${local.pipeline_image_tags["oam-ingest"]}"
        args = [
          "--with-events",
          "--with-360",
          "--gcs-bucket", google_storage_bucket.data.name,
          "--data-version", var.pipeline_data_version,
          "--source-ref", var.pipeline_data_version,
        ]
        resources {
          limits = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_job" "oam_transform" {
  name                = "oam-transform"
  project             = var.project_id
  location            = var.region
  deletion_protection = false
  labels              = local.common_labels

  template {
    template {
      service_account = google_service_account.oam_pipeline_sa.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = "${local.artifact_registry}/oam-transform:${local.pipeline_image_tags["oam-transform"]}"
        args = [
          "--project-id", var.project_id,
          "--region", var.region,
          "--bucket", google_storage_bucket.data.name,
          "--artifacts-bucket", google_storage_bucket.artifacts.name,
          "--data-version", var.pipeline_data_version,
          "--source-ref", var.pipeline_data_version,
        ]
        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_job" "oam_analyse" {
  name                = "oam-analyse"
  project             = var.project_id
  location            = var.region
  deletion_protection = false
  labels              = local.common_labels

  template {
    template {
      service_account = google_service_account.oam_pipeline_sa.email
      max_retries     = 1
      timeout         = "3600s"

      containers {
        image = "${local.artifact_registry}/oam-analyse:${local.pipeline_image_tags["oam-analyse"]}"
        args  = []
        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

# --- IAM: oam-pipeline-sa, scoped to exactly what each job touches ---
#
# No existing IAM-binding precedent for this SA (runtime_identity.tf only creates it);
# scoping decisions below are made fresh, tightened to dataset/bucket/resource level
# wherever GCP's IAM model allows it, matching the task's "match or tighten, never loosen"
# instruction against a from-scratch baseline.

resource "google_bigquery_dataset_iam_member" "pipeline_sa_core_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.datasets["oam_core"].dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_bigquery_dataset_iam_member" "pipeline_sa_features_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.datasets["oam_features"].dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_bigquery_dataset_iam_member" "pipeline_sa_analysis_editor" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.datasets["oam_analysis"].dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

# BigQuery has no dataset-scoped equivalent of "may run a query job" -- job creation is
# inherently a project-level permission in BigQuery's IAM model. This is the narrowest
# role that grants it (does not grant dataEditor/read on datasets beyond what's bound above).
resource "google_project_iam_member" "pipeline_sa_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_storage_bucket_iam_member" "pipeline_sa_data_bucket" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_storage_bucket_iam_member" "pipeline_sa_artifacts_bucket" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}
