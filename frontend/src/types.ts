export type GameStatus = "waiting" | "active" | "finished";

export interface ChatMessage {
  sender: string;
  text: string;
  timestamp: number;
  system: boolean;
  isBot: boolean;
}

export interface RoomPlayer {
  id: string;
  name: string;
  isBot: boolean;
}

export interface RoomState {
  players: RoomPlayer[];
  scores: Record<string, number>;
  status: GameStatus;
  roundEndTime: number | null;
}

export interface ServerEvent {
  type: string;
  state?: RoomState;
  message?: ChatMessage;
  username?: string;
  messageText?: string;
}

export interface MemoryRecord {
  id: number;
  memoryText: string;
  metadata: Record<string, string>;
}
