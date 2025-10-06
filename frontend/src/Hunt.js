// frontend/src/Hunt.js
import React, { useEffect, useState } from "react";

function Hunt() {
    const API_URL = "http://192.168.110.18:8000"; //process.env.REACT_APP_API_URL ||

    const [sessionId, setSessionId] = useState(null);
    const [huntItems, setHuntItems] = useState([]); // [{id,name,description,image_url,found,user_image_url}]
    const [loading, setLoading] = useState(false);
    const [file, setFile] = useState(null);

    // Get or create a session
    useEffect(() => {
        let sid = localStorage.getItem("session_id");
        (async () => {
            if (!sid) {
                const r = await fetch(`${API_URL}/session/start`, { method: "POST" });
                const data = await r.json();
                sid = data.session_id;
                localStorage.setItem("session_id", sid);
            }
            setSessionId(sid);

            // Load items & session progress
            const [itemsRes, progRes] = await Promise.all([
                fetch(`${API_URL}/items`),
                fetch(`${API_URL}/session/${sid}/progress`),
            ]);
            const itemsData = await itemsRes.json();
            const progData = await progRes.json();

            // Merge base items with progress flags/images
            const base = (itemsData.items || []).map((it) => ({
                ...it,
                found: false,
                user_image_url: null,
            }));

            const progressMap = {};
            (progData.items || []).forEach((it) => (progressMap[it.id] = it));

            const merged = base.map((it) => {
                const p = progressMap[it.id];
                return p
                    ? { ...it, found: p.found, user_image_url: p.user_image_url }
                    : it;
            });

            setHuntItems(merged);
        })();
    }, [API_URL]);

    const allFound = huntItems.length > 0 && huntItems.every((i) => i.found);

    const handleUpload = async () => {
        if (!file || !sessionId) return;
        setLoading(true);

        const form = new FormData();
        form.append("file", file);
        form.append("session_id", sessionId);

        try {
            const res = await fetch(`${API_URL}/upload`, {
                method: "POST",
                body: form,
            });
            const data = await res.json();

            if (data.success) {
                // Mark the matched item as found; show server-served user image URL
                setHuntItems((prev) =>
                    prev.map((i) =>
                        i.id === data.item.id
                            ? {
                                ...i,
                                found: true,
                                user_image_url: data.user_image_url, // persists across refresh
                            }
                            : i
                    )
                );
                if (data.completed) {
                    alert("🎉 You completed the hunt!");
                }
            } else {
                alert("❌ No match. Try again!");
            }
        } catch (e) {
            console.error(e);
            alert("⚠️ Upload failed");
        } finally {
            setLoading(false);
            setFile(null);
        }
    };

    return (
        <div style={{ padding: "2rem", fontFamily: "Inter, system-ui, sans-serif" }}>
            <h2>Pinecone Scavenger Hunt</h2>

            <div style={{ marginBottom: "1rem", color: "#555" }}>
                Session: <code>{sessionId || "…"}</code> • Progress:{" "}
                <strong>
                    {huntItems.filter((i) => i.found).length}/{huntItems.length}
                </strong>
            </div>

            <h3>Your Hunt List</h3>
            <ul style={{ listStyle: "none", padding: 0 }}>
                {(huntItems || []).map((item) => (
                    <li
                        key={item.id}
                        style={{
                            marginBottom: "1rem",
                            paddingBottom: "1rem",
                            borderBottom: "1px solid #eee",
                        }}
                    >
                        <div style={{ fontSize: "1rem" }}>
                            {item.found ? "✅" : "❌"} {item.description}
                        </div>

                        {/* Only show images once found */}
                        {item.found && (
                            <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem" }}>
                                {item.user_image_url && (
                                    <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Your Photo</div>
                                        <img
                                            src={`${API_URL}${item.user_image_url}`}
                                            alt="Your upload"
                                            style={{
                                                maxWidth: "160px",
                                                borderRadius: 8,
                                                border: "2px solid #ccc",
                                            }}
                                        />
                                    </div>
                                )}
                                {item.image_url && (
                                    <div>
                                        <div style={{ fontSize: 12, color: "#666" }}>Reference</div>
                                        <img
                                            src={item.image_url}
                                            alt={item.name || ""}
                                            style={{
                                                maxWidth: "160px",
                                                borderRadius: 8,
                                                border: "2px solid #4CAF50",
                                            }}
                                        />
                                    </div>
                                )}
                            </div>
                        )}
                    </li>
                ))}
            </ul>

            {!allFound && (
                <div style={{ marginTop: "1rem" }}>
                    <input
                        type="file"
                        accept="image/*"
                        capture="environment"
                        onChange={(e) => setFile(e.target.files[0])}
                    />
                    <button
                        onClick={handleUpload}
                        disabled={!file || loading}
                        style={{
                            marginLeft: "0.5rem",
                            padding: "0.5rem 1rem",
                            borderRadius: 6,
                            border: "none",
                            background: "#4CAF50",
                            color: "white",
                            cursor: !file || loading ? "default" : "pointer",
                        }}
                    >
                        {loading ? "Checking..." : "Upload"}
                    </button>
                </div>
            )}

            {allFound && (
                <div style={{ marginTop: "1.5rem", color: "#2e7d32" }}>
                    <h2>🎉 Congratulations! You found all the items!</h2>
                </div>
            )}
        </div>
    );
}


export default Hunt;
