"use client"

import React, { useState } from 'react';
import Predict from './components/Predict';
import PredictionResult from './components/PredictionResult';

const HomePage: React.FC = () => {
    const [prediction, setPrediction] = useState<string | null>(null);

    const handlePrediction = (result: string) => {
        setPrediction(result);
    };

    return (
        <div className="max-w-2xl mx-auto">
            <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">Stock Prediction App</h1>
            <div className="bg-white shadow-md rounded-lg p-6 mb-8">
                <Predict onPrediction={handlePrediction} />
            </div>
            {prediction && (
                <div className="bg-white shadow-md rounded-lg p-6">
                    <PredictionResult prediction={prediction} />
                </div>
            )}
        </div>
    );
};

export default HomePage;