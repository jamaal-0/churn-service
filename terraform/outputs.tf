output "api_ecr_repository_url" {
  description = "ECR repository URL for the FastAPI image"
  value       = aws_ecr_repository.api.repository_url
}

output "mlflow_ecr_repository_url" {
  description = "ECR repository URL for the MLflow image"
  value       = aws_ecr_repository.mlflow.repository_url
}

output "initialise_ecr_repository_url" {
  description = "ECR repository URL for the initialisation image"
  value       = aws_ecr_repository.initialise.repository_url
}

output "mlflow_artifact_bucket_name" {
  description = "S3 bucket used for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}