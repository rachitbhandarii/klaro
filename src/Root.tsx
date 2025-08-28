import { Composition } from "remotion";
import { SlidesRenderer } from "./SlidesRenderer";

const videoTopic = process.env.VIDEO_TOPIC;
if (!videoTopic) {
  throw new Error("VIDEO_TOPIC is not set in .env!");
}
export const topic = videoTopic
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-+|-+$/g, "");

console.log("Video topic is:", topic);

const jsonPath: string = `../public/${topic}/final-content/${topic}.json`

const treeDataJson = require(jsonPath);
import { TopicDict } from "./types";

const treeData: TopicDict = treeDataJson as TopicDict;


// Helper: get total duration from audio lengths recursively
const getTotalDurationInFrames = (node: TopicDict, fps: number): number => {
  let totalSeconds = node.audio?.length || 5;

  if (node.subtopics) {
    totalSeconds += node.subtopics.reduce(
      (sum, child) => sum + getTotalDurationInFrames(child, fps) / fps,
      0
    );
  }

  return totalSeconds * fps;
};

export const RemotionRoot: React.FC = () => {
  const fps = 60; // match your project FPS
  const durationInFrames = Math.ceil(getTotalDurationInFrames(treeData, fps));

  return (
    <Composition
      id="Slides"
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
