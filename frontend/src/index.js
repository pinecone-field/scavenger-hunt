import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Hunt from "./Hunt";
import AdminGuard from "./AdminGuard";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
    <Router>
        <Routes>
            <Route path="/" element={<Hunt />} />
            <Route path="/admin" element={<AdminGuard />} />
        </Routes>
    </Router>
);
