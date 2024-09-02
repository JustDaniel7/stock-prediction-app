import React from 'react';

interface PredictionResultProps {
    prediction: string | null;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction }) => {
    return (
        <div className="text-center">
            <h2 className="text-2xl font-semibold mb-2 text-gray-800">Prediction Result</h2>
            {prediction ? (
                <p className="text-xl text-blue-600 font-medium">{prediction}</p>
            ) : (
                <p className="text-xl text-gray-500">No prediction available yet.</p>
            )}
        </div>
    );
};

export default PredictionResult;