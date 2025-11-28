# AI API v4 Implementation Documentation

## Overview

This document describes the implementation of the AI API v4 integration for the OmniMer Health ecosystem, supporting both `recommend` and `evaluate` workflows using the personal model v4.

## API Endpoints

### Base URL
```
/api/v1/ai/v4
```

### Endpoints

1. **POST /recommend** - Get AI-powered exercise recommendations
2. **POST /evaluate** - Evaluate completed workout session
3. **POST /process** - Generic AI service processor
4. **GET /health** - Health check for AI service
5. **GET /info** - Get AI service information
6. **POST /recommend/legacy** - Legacy RAG-based recommendation

## Request/Response Schemas

### 1. Recommend Endpoint

#### Request
```typescript
POST /api/v1/ai/v4/recommend
Content-Type: application/json
Authorization: Bearer <token>

{
  "healthProfile": {
    "gender": "Male",
    "age": 25,
    "height": 175,
    "weight": 70,
    "bmi": 22.8,
    "bodyFatPercentage": 15,
    "activityLevel": 3,
    "experienceLevel": "Intermediate",
    "workoutFrequency": 4,
    "restingHeartRate": 60,
    "healthStatus": {
      "injuries": ["Knee"]
    }
  },
  "goals": [
    {
      "goalType": "MuscleGain",
      "targetMetric": []
    }
  ],
  "exercises": [
    {
      "exerciseId": "64f8a...",
      "exerciseName": "Bench Press"
    }
  ],
  "k": 5
}
```

#### Response
```typescript
{
  "success": true,
  "message": "Exercise recommendations generated successfully",
  "data": {
    "exercises": [
      {
        "name": "Barbell Bench Press",
        "sets": [
          {
            "reps": 8,
            "kg": 60,
            "minRest": 90
          },
          {
            "reps": 8,
            "kg": 60,
            "minRest": 90
          },
          {
            "reps": 8,
            "kg": 60,
            "minRest": 90
          },
          {
            "reps": 8,
            "kg": 60,
            "minRest": 90
          }
        ]
      },
      {
        "name": "Treadmill",
        "sets": [
          {
            "distance": 0.78
          }
        ]
      }
    ]
  }
}
```

### 2. Evaluate Endpoint

#### Request
```typescript
POST /api/v1/ai/v4/evaluate
Content-Type: application/json
Authorization: Bearer <token>

{
  "healthProfile": {
    "gender": "Male",
    "age": 25,
    "height": 175,
    "weight": 70,
    "bmi": 22.8,
    "bodyFatPercentage": 15,
    "activityLevel": 3,
    "experienceLevel": "Intermediate",
    "workoutFrequency": 4,
    "restingHeartRate": 60,
    "healthStatus": {
      "injuries": ["Knee"]
    }
  },
  "timeStart": "2025-11-28T14:00:00.000Z",
  "notes": "Buổi tập ngực và tim mạch.",
  "workoutDetail": [
    {
      "_id": "60c72b1f90c42d0015f6c8d5",
      "exerciseId": "60c72b1f90c42d0015f6c8d6",
      "type": "reps",
      "sets": [
        {
          "_id": "60c72b1f90c42d0015f6c8d7",
          "setOrder": 1,
          "reps": 8,
          "weight": 60,
          "restAfterSetSeconds": 90,
          "notes": "",
          "done": true
        }
      ],
      "durationMin": 15,
      "deviceData": {
        "heartRateAvg": 115,
        "heartRateMax": 130,
        "caloriesBurned": 100
      }
    }
  ],
  "summary": {
    "heartRateAvgAllWorkout": 131,
    "heartRateMaxAllWorkout": 160,
    "totalSets": 6,
    "totalReps": 32,
    "totalWeight": 240,
    "totalDuration": 1500,
    "totalCalories": 210,
    "totalDistance": 780
  }
}
```

#### Response
```typescript
{
  "success": true,
  "message": "Workout evaluation completed successfully",
  "data": {
    "results": [
      {
        "exerciseName": "Barbell Bench Press",
        "intensityScore": 4,
        "suitability": 0.85
      },
      {
        "exerciseName": "Treadmill",
        "intensityScore": 3,
        "suitability": 0.8
      }
    ]
  }
}
```

### 3. Generic Process Endpoint

#### Request
```typescript
POST /api/v1/ai/v4/process
Content-Type: application/json
Authorization: Bearer <token>

{
  "type": "recommend", // or "evaluate"
  "data": {
    // Use the same data structures as specific endpoints above
  }
}
```

#### Response
```typescript
{
  "success": true,
  "data": {
    // Response data based on the request type
  }
}
```

## Implementation Details

### Files Created/Modified

1. **Entities**: `src/domain/entities/AI.entity.ts`
   - Defines all TypeScript interfaces for AI API v4
   - Includes both recommend and evaluate workflow types
   - Maintains backward compatibility with RAG entities

2. **Service**: `src/domain/services/AI.service.ts`
   - Enhanced existing AI service with v4 support
   - Added `recommendExercisesV4()` method
   - Added `evaluateWorkout()` method
   - Added `processRequest()` generic method
   - Added utility methods: `healthCheck()`, `getServiceInfo()`

3. **Controller**: `src/domain/controllers/AI.controller.ts`
   - New controller for AI v4 endpoints
   - Handles request validation and error handling
   - Implements all AI workflow methods

4. **Routes**: `src/domain/routes/AI.routes.ts`
   - Route definitions for AI v4 endpoints
   - Authentication middleware integration
   - Service dependency injection

5. **Route Integration**: `src/domain/routes/index.ts`
   - Added AI v4 routes to main application

### AI Service Configuration

The AI service requires the following environment variable:

```bash
AI_API=http://localhost:8000  # AI server FastAPI endpoint
```

### Authentication

All endpoints (except `/health` and `/info`) require JWT authentication:

```typescript
Authorization: Bearer <jwt_token>
```

## Usage Examples

### Frontend Integration (React/TypeScript)

```typescript
// Service class for AI API calls
class AIService {
  private baseUrl = '/api/v1/ai/v4';

  async recommendExercises(request: IRecommendInput): Promise<IRecommendOutput> {
    const response = await fetch(`${this.baseUrl}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${getToken()}`
      },
      body: JSON.stringify(request)
    });

    return response.json();
  }

  async evaluateWorkout(request: IEvaluateInput): Promise<IEvaluateOutput> {
    const response = await fetch(`${this.baseUrl}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${getToken()}`
      },
      body: JSON.stringify(request)
    });

    return response.json();
  }
}
```

### Mobile App Integration (Flutter/Dart)

```dart
class AIService {
  final String _baseUrl = '/api/v1/ai/v4';

  Future<Map<String, dynamic>> recommendExercises(Map<String, dynamic> request) async {
    final token = await getAuthToken();

    final response = await http.post(
      Uri.parse('$_baseUrl/recommend'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode(request),
    );

    return jsonDecode(response.body);
  }

  Future<Map<String, dynamic>> evaluateWorkout(Map<String, dynamic> request) async {
    final token = await getAuthToken();

    final response = await http.post(
      Uri.parse('$_baseUrl/evaluate'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
      body: jsonEncode(request),
    );

    return jsonDecode(response.body);
  }
}
```

## Error Handling

### Standard Error Response

```typescript
{
  "success": false,
  "message": "Error description",
  "data": null
}
```

### Common Error Codes

- **400 Bad Request**: Missing required fields or invalid data format
- **401 Unauthorized**: Invalid or missing JWT token
- **500 Internal Server Error**: AI service unavailable or internal error
- **503 Service Unavailable**: AI service health check failed

## AI Model Integration

The backend integrates with the AI server's personal model v4, which uses:

- **Two-Branch Architecture**:
  - Branch A: Intensity prediction (RPE 1-10)
  - Branch B: Suitability prediction (0-1)

- **Feature Engineering**:
  - 28 input features from user profile and exercise metadata
  - Biometric data from health devices
  - Historical workout data

- **Evaluation Metrics**:
  - Intensity RMSE: 0.21
  - Intensity R²: 0.993
  - Suitability Accuracy: 98.7%
  - AUC-ROC: 0.999

## Deployment Notes

1. **Environment Setup**: Ensure `AI_API` environment variable points to the FastAPI server
2. **Health Monitoring**: Use `/api/v1/ai/v4/health` endpoint for monitoring
3. **Rate Limiting**: Consider implementing rate limiting for AI endpoints
4. **Caching**: Cache AI responses when appropriate to reduce server load
5. **Fallback**: Implement fallback logic when AI service is unavailable

## Testing

### Health Check
```bash
curl -X GET http://localhost:3000/api/v1/ai/v4/health
```

### Service Info
```bash
curl -X GET http://localhost:3000/api/v1/ai/v4/info
```

### Recommend (with auth)
```bash
curl -X POST http://localhost:3000/api/v1/ai/v4/recommend \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d @recommend_input.json
```

### Evaluate (with auth)
```bash
curl -X POST http://localhost:3000/api/v1/ai/v4/evaluate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d @evaluate_input.json
```

## Future Enhancements

1. **Real-time Recommendations**: WebSocket integration for live workout adjustments
2. **Batch Processing**: Support for evaluating multiple workouts
3. **Custom Models**: Support for user-specific trained models
4. **A/B Testing**: Framework for testing different AI model versions
5. **Analytics Dashboard**: AI performance monitoring and usage analytics