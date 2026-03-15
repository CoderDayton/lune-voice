export interface HistoryItem {
  time: string;
  text: string;
  words: number;
  seconds: number;
  mode: string;
}

export interface HistoryGroup {
  date: string;
  items: HistoryItem[];
}

export const historyGroups: HistoryGroup[] = [
  {
    date: "Today",
    items: [
      {
        time: "9:01 AM",
        text: "I want you to navigate the website, particularly the blog page, and I want you to review the blog page and come up with improvements to the UI for the blog page as well as the blog ID pages. It needs to be very modern and unique.",
        words: 47,
        seconds: 15,
        mode: "Balanced",
      },
      {
        time: "8:44 AM",
        text: "Okay, execute the plan.",
        words: 4,
        seconds: 2,
        mode: "Quick paste",
      },
      {
        time: "8:22 AM",
        text: "To learn more, visit each product page.",
        words: 32,
        seconds: 11,
        mode: "Compact",
      },
      {
        time: "7:55 AM",
        text: "Please generate a summary of the current session and export it to markdown.",
        words: 14,
        seconds: 5,
        mode: "Balanced",
      },
    ],
  },
  {
    date: "Yesterday",
    items: [
      {
        time: "11:32 PM",
        text: "Fix the authentication middleware to handle expired tokens gracefully and return proper error codes.",
        words: 16,
        seconds: 6,
        mode: "Balanced",
      },
      {
        time: "3:14 PM",
        text: "Schedule a meeting with the design team for next Monday at two PM.",
        words: 14,
        seconds: 5,
        mode: "Balanced",
      },
      {
        time: "10:07 AM",
        text: "Open the project dashboard and filter by active tasks assigned to me.",
        words: 13,
        seconds: 5,
        mode: "Compact",
      },
    ],
  },
  {
    date: "Mar 9",
    items: [
      {
        time: "8:30 PM",
        text: "Wow, this is kind of amazing.",
        words: 6,
        seconds: 2,
        mode: "Balanced",
      },
      {
        time: "2:45 PM",
        text: "Transcribe anything locally without sending audio to the cloud.",
        words: 11,
        seconds: 4,
        mode: "Balanced",
      },
    ],
  },
];
