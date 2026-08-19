resource "google_service_account" "oam_pipeline_sa" {
  account_id   = "oam-pipeline-sa"
  project      = var.project_id
  display_name = "OAM Pipeline Service Account"
  description  = "Service account for OAM pipeline runtime (Cloud Run jobs for ingestion, feature builds, training, evaluation, promotion, serving builds)."

  deletion_policy = "PREVENT"

  lifecycle {
    prevent_destroy = true
  }
}
