from fastapi import APIRouter, HTTPException
from src.model.predict import predict_next_day

router = APIRouter()

@router.get("/predict/{company_code}")
async def get_prediction(company_code: str):
    try:
        prediction = predict_next_day(company_code)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Model or data not found for the given company")
        return {"company_code": company_code, "predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
