import React, { useState } from 'react';

interface PredictProps {
    onPrediction: (result: string) => void;
}

const Predict: React.FC<PredictProps> = ({ onPrediction }) => {
    const [companyCode, setCompanyCode] = useState<string>('');
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict/${companyCode}`);
            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            onPrediction(`Predicted Price: $${data.predicted_price.toFixed(2)}`);
        } catch (error: any) {
            setError(error.message);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label htmlFor="companyCode" className="block text-sm font-medium text-gray-700 mb-1">
                        Company Code
                    </label>
                    <input
                        id="companyCode"
                        type="text"
                        value={companyCode}
                        onChange={(e) => setCompanyCode(e.target.value)}
                        placeholder="Enter company code"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                </div>
                <button
                    type="submit"
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-150 ease-in-out"
                >
                    Get Prediction
                </button>
            </form>
            {error && <p className="mt-4 text-red-600">{error}</p>}
        </div>
    );
};

export default Predict;