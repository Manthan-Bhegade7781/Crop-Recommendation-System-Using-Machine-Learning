import React from "react";

export default function ResultCard({ result }) {
  if (!result) return null;

  return (
    <div className="mt-8 bg-green-50 border border-green-200 p-6 rounded-xl text-center">
      <h3 className="text-xl font-semibold text-green-700 mb-2">
        🌾 Recommended Crop
      </h3>
      <p className="text-3xl font-bold text-gray-800 mb-2">
        {result.recommendation}
      </p>

      {result.probability && (
        <p className="text-sm text-gray-600">
          Confidence: {(result.probability * 100).toFixed(1)}%
        </p>
      )}

      {result.alternatives?.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium text-gray-700">Alternative Crops</h4>
          <ul className="list-disc list-inside text-gray-600 text-sm">
            {result.alternatives.map((alt, i) => (
              <li key={i}>{alt}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
