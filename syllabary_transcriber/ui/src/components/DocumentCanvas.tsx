import React, { useEffect, useRef } from 'react';

interface DocumentCanvasProps {
  text: string;
  onChangeText: (newText: string) => void;
}

export const DocumentCanvas: React.FC<DocumentCanvasProps> = ({ text, onChangeText }) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
    }
  }, [text]);

  return (
    <div className="document-canvas-container">
      <textarea
        ref={textareaRef}
        className="document-canvas-textarea"
        value={text}
        onChange={(e) => onChangeText(e.target.value)}
        placeholder="Transcribed Cherokee syllabary text will appear here..."
        spellCheck={false}
      />
    </div>
  );
};

export default DocumentCanvas;
