import React, { useMemo } from "react";
import { useVideoConfig, AbsoluteFill, Sequence } from "remotion";
import { Slide } from "./Slide";
import { TopicDict } from "./types";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { slide as slideTransition } from "@remotion/transitions/slide";

type Props = {
  tree: TopicDict;
};

// DFS traversal to flatten tree
const dfsSlides = (node: TopicDict): TopicDict[] => {
  const slides: TopicDict[] = [];
  const stack: TopicDict[] = [node];

  while (stack.length > 0) {
    const curr = stack.pop()!;
    slides.push(curr);

    if (curr.subtopics) {
      for (let i = curr.subtopics.length - 1; i >= 0; i--) {
        stack.push(curr.subtopics[i]);
      }
    }
  }

  return slides;
};

export const SlidesRenderer: React.FC<Props> = ({ tree }) => {
  const { fps } = useVideoConfig();

  // Flatten slides in DFS order
  const allSlides = useMemo(() => dfsSlides(tree), [tree]);

  return (
    <AbsoluteFill>
      {allSlides.map((slide, index) => {
        const audioLength = slide.audio?.length || 5; // fallback 5s
        const durationInFrames = Math.ceil(audioLength * fps);

        const fromFrame = allSlides
          .slice(0, index)
          .reduce((acc, s) => acc + ((s.audio?.length || 5) * fps), 0);

        return (
          <Sequence
            key={index}
            from={fromFrame}
            durationInFrames={durationInFrames}
          >
            <TransitionSeries>
              <TransitionSeries.Sequence durationInFrames={durationInFrames}>
                <Slide handle={slide} index={index} />
              </TransitionSeries.Sequence>

              <TransitionSeries.Transition
                presentation={slideTransition({ direction: "from-right" })}
                timing={linearTiming({ durationInFrames: 30 })}
              />
            </TransitionSeries>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
