import React from 'react';

interface PredictionResultProps {
    prediction: string;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction }) => {
    return (
        <div className="text-center">
            <h2 className="text-2xl font-semibold mb-2 text-gray-800">Prediction Result</h2>
            <p className="text-xl text-blue-600 font-medium">{prediction}</p>
        </div>
    );
};

export default PredictionResult;