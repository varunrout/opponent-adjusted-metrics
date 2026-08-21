# Cloud Workflows orchestration: chains oam-ingest -> oam-transform -> oam-analyse.
#
# Manual invocation only. No Cloud Scheduler trigger is provisioned here -- the task
# explicitly requires confirming with the user first whether StatsBomb's open data changes
# on a cadence that justifies automatic re-ingestion before adding one; that confirmation
# has not happened, so this stays manual (`gcloud workflows run oam-pipeline ...` or Console).
#
# No 4th "model" job/step exists in the referenced workflow source
# (infra/workflows/oam_pipeline.yaml) or anywhere in this file.

resource "google_project_service" "workflows" {
  service            = "workflows.googleapis.com"
  project            = var.project_id
  disable_on_destroy = false
  deletion_policy    = "PREVENT"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_workflows_workflow" "oam_pipeline" {
  name            = "oam-pipeline"
  project         = var.project_id
  region          = var.region
  description     = "Chains oam-ingest -> oam-transform -> oam-analyse on completion. Manual invocation only. No model/training step."
  service_account = google_service_account.oam_pipeline_sa.id
  source_contents = file("${path.module}/../workflows/oam_pipeline.yaml")
  labels          = local.common_labels

  user_env_vars = {
    PIPELINE_PROJECT_ID = var.project_id
  }

  depends_on = [google_project_service.workflows]
}

# The workflow executes as oam-pipeline-sa (reused, no new identity) and needs permission
# to start + poll each job's executions. Per-job roles/run.invoker (below) covers *starting*
# an execution, scoped to each specific job resource. It does NOT cover polling: real
# end-to-end validation hit a live 403 on `run.operations.get` when the Workflows Cloud Run
# connector polled the async execution's long-running operation
# (projects/*/locations/*/operations/*) -- that resource type is not nested under any single
# job, so IAM cannot be scoped to it per-job; it requires a project/location-level grant.
# roles/run.developer is the narrowest predefined role that includes run.operations.get
# alongside jobs.run/executions.get, so it's granted at project level here as a deliberate,
# evidence-backed exception to per-resource scoping -- not a default choice.
resource "google_cloud_run_v2_job_iam_member" "pipeline_sa_invoke_ingest" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.oam_ingest.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_cloud_run_v2_job_iam_member" "pipeline_sa_invoke_transform" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.oam_transform.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_cloud_run_v2_job_iam_member" "pipeline_sa_invoke_analyse" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.oam_analyse.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}

resource "google_project_iam_member" "pipeline_sa_run_developer" {
  project = var.project_id
  role    = "roles/run.developer"
  member  = "serviceAccount:${google_service_account.oam_pipeline_sa.email}"
}
