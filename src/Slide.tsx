import React from "react";
import {
  spring,
  interpolate,
  Audio,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
  staticFile
} from "remotion";
import { TopicDict } from "./types";
import { topic } from "./Root";

type SlideProps = {
  handle: TopicDict;
  index: number;
};

export const Slide: React.FC<SlideProps> = ({ handle, index }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const audioUrl = staticFile(`/${ topic }/audio/${index + 1}.mp3`);

  // Smooth spring scaling for the title
  const scale = spring({
    fps,
    frame,
    config: {
      damping: 200,
      stiffness: 100,
      mass: 0.5,
    },
  });

  // Fade-in animation for points
  const opacity = interpolate(frame, [0, 60], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(135deg, #6a0683ef 0%, #112240 100%)",
        color: "white",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Inter, sans-serif",
        padding: "40px 20px",
      }}
    >
      {/* Glassy overlay card */}
      <div
        style={{
          background: "rgba(5, 17, 39, 0.6)",
          backdropFilter: "blur(14px) saturate(160%)",
          WebkitBackdropFilter: "blur(14px) saturate(160%)",
          borderRadius: "24px",
          padding: "40px 80px",
          boxShadow: "0 8px 40px rgba(0,0,0,0.7)",
          maxWidth: "80%",
          border: "1px solid rgba(51, 0, 66, 0.15)",
        }}
      >
        {/* Title */}
        <h1
          style={{
            marginBottom: "30px",
            fontSize: "3.8rem",
            fontWeight: 800,
            letterSpacing: "-1px",
            background: "linear-gradient(90deg, #db83ecff, rgba(240, 204, 243, 1))",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            transform: `scale(${scale})`,
          }}
        >
          {handle.topic}
        </h1>

        {/* Bullet Points */}
        <ul
          style={{
            listStyle: "none",
            padding: 0,
            margin: 0,
          }}
        >
          {handle.slide.map((s, i) => (
            <li
              key={i}
              style={{
                margin: "14px 0",
                fontSize: "1.9rem",
                fontWeight: 400,
                opacity,
                transform: `translateY(${20 - opacity * 20}px)`,
                lineHeight: "1.5",
                background: "rgba(241, 159, 224, 0.08)",
                padding: "12px 22px",
                borderRadius: "14px",
                border: "1px solid rgba(0,229,255,0.25)",
                boxShadow: "inset 0 1px 6px rgba(0,229,255,0.15)",
                transition: "all 0.5s ease",
              }}
            >
              → {s.point}
            </li>
          ))}
        </ul>
      </div>

      {/* Audio */}
      {<Audio src={audioUrl} />}
    </AbsoluteFill>
  );
};
