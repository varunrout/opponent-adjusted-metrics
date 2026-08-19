// Enable minimal runtime APIs required for pipeline runtime foundation

resource "google_project_service" "artifactregistry" {
  service            = "artifactregistry.googleapis.com"
  project            = var.project_id
  disable_on_destroy = false
  deletion_policy    = "PREVENT"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_project_service" "run" {
  service            = "run.googleapis.com"
  project            = var.project_id
  disable_on_destroy = false
  deletion_policy    = "PREVENT"

  lifecycle {
    prevent_destroy = true
  }
}
