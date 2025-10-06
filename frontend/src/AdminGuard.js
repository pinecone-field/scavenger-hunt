// src/AdminGuard.js
import React, { useState } from "react";
import Admin from "./admin/Admin";

const AdminGuard = () => {
    const [authed, setAuthed] = useState(localStorage.getItem("adminAuthed") === "true");
    const [password, setPassword] = useState("");

    const handleLogin = () => {
        // In real life, check against backend. For now hardcode a password.
        if (password === "PineconeAWS2025!") {
            localStorage.setItem("adminAuthed", "true");
            setAuthed(true);
        } else {
            alert("Wrong password");
        }
    };

    if (!authed) {
        return (
            <div style={{ padding: "2rem" }}>
                <h2>Admin Login</h2>
                <input
                    type="password"
                    value={password}
                    placeholder="Enter admin password"
                    onChange={(e) => setPassword(e.target.value)}
                />
                <button onClick={handleLogin}>Login</button>
            </div>
        );
    }

    return <Admin />;
};

export default AdminGuard;
