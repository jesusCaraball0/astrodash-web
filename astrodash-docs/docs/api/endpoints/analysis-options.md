---
sidebar_position: 4
---

# Analysis Options

Get available SN types and age bins for analysis.

## Endpoint

```
GET /api/analysis-options
```

## Description

Returns the list of available supernova types and their corresponding age bins for use in spectrum processing and classification.

## Request

No parameters required.

## Response

### Success Response
**Status Code:** `200 OK`

```json
{
  "sn_types": ["Ia-norm", "Ib-norm", ...],
  "age_bins_by_type": {
    "Ia-norm": ["-10 to -6", "-6 to -2", ...],
    ...
  }
}
```

### Error Response
**Status Code:** `500 Internal Server Error`

```json
{
  "detail": "Fetching analysis options failed"
}
```

## Example

### cURL
```bash
curl -X GET "http://localhost:5000/api/analysis-options"
```

### Python
```python
import requests
response = requests.get('http://localhost:5000/api/analysis-options')
print(response.json())
```

## Notes
- Use these SN types and age bins as valid values for other endpoints (e.g., `/api/template-spectrum`).
