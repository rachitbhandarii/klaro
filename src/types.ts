export type SlideContent = {
  point: string;
  start_time: number;
  end_time: number;
};

export type TopicDict = {
  topic: string;
  content: string;
  questions: string[];
  slide: SlideContent[];
  audio: {
    audio_base_64: string;
    length: number;
    alignment: {
      characters: string[];
      character_start_times_seconds: number[];
      character_end_times_seconds: number[];
    };
    normalized_alignment: {
      characters: string[];
      character_start_times_seconds: number[];
      character_end_times_seconds: number[];
    };
  };
  subtopics?: TopicDict[];
};

const videoTopic = process.env.VIDEO_TOPIC;
if (!videoTopic) {
  throw new Error("VIDEO_TOPIC is not set in .env!");
}
export const topic = videoTopic
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-+|-+$/g, "");

console.log("Video topic is:", topic);