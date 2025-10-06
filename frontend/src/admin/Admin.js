import React, { useEffect, useState } from "react";

export default function Admin() {
    const API_URL = process.env.REACT_APP_API_URL || "http://192.168.110.18:8000";
    const [winners, setWinners] = useState([]);

    useEffect(() => {
        (async () => {
            const res = await fetch(`${API_URL}/admin/winners`);
            const data = await res.json();
            setWinners(data.winners || []);
        })();
    }, [API_URL]);

    return (
        <div style={{ padding: "2rem" }}>
            <h2>🏆 Scavenger Hunt Winners</h2>
            {winners.map((w) => (
                <div
                    key={w.session_id}
                    style={{
                        marginBottom: "2rem",
                        padding: "1rem",
                        border: "1px solid #ddd",
                        borderRadius: 8,
                    }}
                >
                    <h3>Session: {w.session_id}</h3>
                    <p>
                        Completed:{" "}
                        {w.completed_at ? new Date(w.completed_at).toLocaleString() : "—"}
                    </p>
                    <ul>
                        {w.items.map((it) => (
                            <li key={it.item_id} style={{ marginBottom: "0.5rem" }}>
                                <strong>{it.name}</strong> <br />
                                Score: {it.pinecone_score.toFixed(2)} <br />
                                LLM: {it.llm_reply} <br />
                                <img
                                    src={`${API_URL}${it.user_image_url}`}
                                    alt={it.name}
                                    style={{ maxWidth: 150, marginTop: "0.5rem" }}
                                />
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}
