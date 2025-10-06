import React from "react";
import { Routes, Route } from "react-router-dom";
import Hunt from "./Hunt";
import Admin from "./admin/Admin"; // <-- note the path

export default function App() {
    return (
        <Routes>
            <Route path="/" element={<Hunt />} />
            <Route path="/admin" element={<Admin />} />
        </Routes>
    );
}
