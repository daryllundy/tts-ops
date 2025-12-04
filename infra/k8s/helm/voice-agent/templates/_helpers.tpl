{{/*
Expand the name of the chart.
*/}}
{{- define "voice-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "voice-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "voice-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "voice-agent.labels" -}}
helm.sh/chart: {{ include "voice-agent.chart" . }}
{{ include "voice-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "voice-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "voice-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
TTS service labels
*/}}
{{- define "voice-agent.tts.labels" -}}
{{ include "voice-agent.labels" . }}
app.kubernetes.io/component: tts
{{- end }}

{{/*
TTS selector labels
*/}}
{{- define "voice-agent.tts.selectorLabels" -}}
{{ include "voice-agent.selectorLabels" . }}
app.kubernetes.io/component: tts
{{- end }}

{{/*
Agent service labels
*/}}
{{- define "voice-agent.agent.labels" -}}
{{ include "voice-agent.labels" . }}
app.kubernetes.io/component: agent
{{- end }}

{{/*
Agent selector labels
*/}}
{{- define "voice-agent.agent.selectorLabels" -}}
{{ include "voice-agent.selectorLabels" . }}
app.kubernetes.io/component: agent
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "voice-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "voice-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
TTS service internal URL
*/}}
{{- define "voice-agent.ttsUrl" -}}
http://{{ include "voice-agent.fullname" . }}-tts:{{ .Values.ttsService.service.port }}
{{- end }}
