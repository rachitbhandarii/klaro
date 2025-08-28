import React, { useMemo } from "react";
import { useVideoConfig, AbsoluteFill } from "remotion";
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
  const allSlides = useMemo(() => dfsSlides(tree), [tree]);

  return (
    <AbsoluteFill>
      <TransitionSeries>
        {allSlides.map((slide, index) => {
          const audioLength = slide.audio?.length || 5;
          const durationInFrames = Math.ceil((audioLength + 2) * fps);

          return (
            <React.Fragment key={index}>
              <TransitionSeries.Sequence durationInFrames={durationInFrames}>
                <Slide handle={slide} index={index} />
              </TransitionSeries.Sequence>

              {/* Add transition only if there is a next slide */}
              {index < allSlides.length - 1 && (
                <TransitionSeries.Transition
                  presentation={slideTransition({ direction: "from-right" })}
                  timing={linearTiming({ durationInFrames: 30 })}
                />
              )}
            </React.Fragment>
          );
        })}
      </TransitionSeries>
    </AbsoluteFill>
  );
};
