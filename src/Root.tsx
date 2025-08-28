import { Composition, staticFile, delayRender, continueRender } from "remotion";
import { SlidesRenderer } from "./SlidesRenderer";
import { TopicDict, topic } from "./types";
import React, { useEffect, useState } from "react";
import { BlankComponent } from "./BlankComponent";

const getTotalDurationInFrames = (node: TopicDict, fps: number): number => {
  let totalSeconds = node.audio?.length + 2 || 5;

  if (!node) {
    return 0;
  }

  if (node.subtopics) {
    totalSeconds += node.subtopics.reduce(
      (sum, child) => sum + getTotalDurationInFrames(child, fps) / fps,
      0
    );
  }

  return totalSeconds * fps;
};

export const RemotionRoot: React.FC = () => {
  const fps = 60;
  const [treeData, setTreeData] = useState<TopicDict | null>(null);

  // This handle prevents rendering until data is fetched
  const [handle] = useState(() => delayRender());

  useEffect(() => {
    const fetchData = async () => {
      try {
        if (typeof window === "undefined") {
          // Node.js render mode
          const data = require(`../public/${topic}/final-content/${topic}.json`) as TopicDict;
          setTreeData(data);
          continueRender(handle);
        } else {
          // Dev (browser)
          const jsonUrl = staticFile(`/${topic}/final-content/${topic}.json`);
          const response = await fetch(jsonUrl);
          const data = (await response.json()) as TopicDict;
          setTreeData(data);
          continueRender(handle);
        }
      } catch (err) {
        console.error("Failed to load JSON:", err);
        continueRender(handle);
      }
    };

    fetchData();
  }, [handle]);

  if (!treeData) {
    return (
      <Composition
        id="empty"
        component={BlankComponent}
        durationInFrames={1}
        fps={fps}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
    );
  }

  const durationInFrames = Math.ceil(getTotalDurationInFrames(treeData, fps));

  return (
    <Composition
      id={topic}
      component={SlidesRenderer}
      durationInFrames={durationInFrames}
      fps={fps}
      width={1920}
      height={1080}
      defaultProps={{
        tree: treeData,
      }}
    />
  );
};
