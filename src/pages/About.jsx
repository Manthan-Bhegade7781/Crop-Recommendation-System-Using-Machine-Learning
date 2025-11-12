import React from "react";

export default function About() {
  return (
    <section className="py-10 px-6 bg-green-50 min-h-screen">
      <div className="max-w-4xl mx-auto bg-white p-8 rounded-2xl shadow-md">
        <h1 className="text-3xl font-bold text-green-700 mb-4 text-center">
          🌱 About Crop Recommendation System
        </h1>

        <p className="text-gray-700 mb-6 text-justify">
          The Crop Recommendation System is an intelligent web application designed to assist 
          farmers and agricultural enthusiasts in making informed decisions about what crop 
          to cultivate based on their soil nutrients, weather, and environmental conditions.
        </p>

        <h2 className="text-2xl font-semibold text-green-700 mb-2">🎯 Objective</h2>
        <p className="text-gray-700 mb-6 text-justify">
          Our goal is to leverage machine learning algorithms to provide accurate and data-driven
          recommendations for farmers, enhancing productivity and sustainable farming practices.
        </p>

        <h2 className="text-2xl font-semibold text-green-700 mb-2">🧠 Technology Stack</h2>
        <ul className="list-disc list-inside text-gray-700 mb-6">
          <li><b>Frontend:</b> React.js + Tailwind CSS</li>
          <li><b>Backend:</b> Flask (Python)</li>
          <li><b>Machine Learning:</b> Scikit-learn (Random Forest, SVM, XGBoost, etc.)</li>
          <li><b>Data:</b> Real-world soil & weather datasets</li>
        </ul>

        <h2 className="text-2xl font-semibold text-green-700 mb-2">🌾 Features</h2>
        <ul className="list-disc list-inside text-gray-700">
          <li>Predicts the most suitable crop for given soil and weather parameters.</li>
          <li>Displays prediction confidence and alternative crop suggestions.</li>
          <li>Clean and responsive web UI using React & Tailwind.</li>
        </ul>
      </div>
    </section>
  );
}
