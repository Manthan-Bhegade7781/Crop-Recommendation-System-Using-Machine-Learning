import React, { useState } from "react";
import ResultCard from "./ResultCard";

export default function CropForm() {
  const [form, setForm] = useState({
    N: "", P: "", K: "", temperature: "", humidity: "", ph: "", rainfall: ""
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          N: +form.N, P: +form.P, K: +form.K,
          temperature: +form.temperature,
          humidity: +form.humidity,
          ph: +form.ph,
          rainfall: +form.rainfall,
        }),
      });

      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("⚠️ Could not connect to Flask. Make sure the backend is running!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-2xl p-8 max-w-3xl mx-auto mt-10">
      <h2 className="text-2xl font-bold text-green-700 mb-6 text-center">
        Enter Your Soil and Weather Details
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {["N", "P", "K"].map((field) => (
            <input
              key={field}
              name={field}
              value={form[field]}
              onChange={handleChange}
              placeholder={field}
              required
              className="p-3 border rounded-md focus:ring focus:ring-green-200"
              type="number"
            />
          ))}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          {["temperature", "humidity", "ph", "rainfall"].map((field) => (
            <input
              key={field}
              name={field}
              value={form[field]}
              onChange={handleChange}
              placeholder={field}
              required
              className="p-3 border rounded-md focus:ring focus:ring-green-200"
              type="number"
              step="0.1"
            />
          ))}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition"
        >
          {loading ? "Predicting..." : "Recommend Crop"}
        </button>

        {error && <p className="text-red-600 mt-3 text-center">{error}</p>}
      </form>

      <ResultCard result={result} />
    </div>
  );
}
