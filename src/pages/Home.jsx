import React from "react";
import CropForm from "../components/CropForm";

export default function Home() {
  return (
    <section className="py-10 px-4">
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-green-700">
          Welcome to Smart Crop Advisor 🌱
        </h1>
        <p className="text-gray-600 mt-2">
          Enter your soil and weather details below to get the best crop recommendation.
        </p>
      </div>
      <CropForm />
    </section>
  );
}
