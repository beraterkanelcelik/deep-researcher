import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HITLWidget } from "@/components/HITLWidgets";
import type { HITLPayload, ResearchReport } from "@/lib/store";

const mockReport: ResearchReport = {
  title: "Test Report",
  summary: "A test summary about AI research",
  key_findings: [
    {
      insight: "AI is advancing rapidly",
      evidence: "Multiple studies confirm",
      sources: ["https://example.com"],
    },
    {
      insight: "Safety research is critical",
      evidence: "Industry consensus",
      sources: ["https://example2.com"],
    },
  ],
  sources: ["https://example.com", "https://example2.com"],
  tags: ["AI", "safety"],
  methodology: "Web search synthesis",
};

describe("HITLWidget dispatcher", () => {
  it("routes to CheckboxWidget for checkbox type", () => {
    const payload: HITLPayload = {
      hitl_type: "checkbox",
      title: "Select Topics",
      message: "Choose topics",
      options: [
        { id: "t1", label: "Topic 1", description: "Desc 1", selected: false },
        { id: "t2", label: "Topic 2", description: "Desc 2", selected: false },
      ],
      report: null,
    };
    render(<HITLWidget payload={payload} onResume={vi.fn()} />);
    expect(screen.getByText("Select Topics")).toBeInTheDocument();
    expect(screen.getByText("Topic 1")).toBeInTheDocument();
    expect(screen.getByText("Topic 2")).toBeInTheDocument();
  });

  it("routes to YesNoWidget for yes_no type", () => {
    const payload: HITLPayload = {
      hitl_type: "yes_no",
      title: "Confirm Action",
      message: "Are you sure?",
      options: [],
      report: null,
    };
    render(<HITLWidget payload={payload} onResume={vi.fn()} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });
});

describe("CheckboxWidget", () => {
  const checkboxPayload: HITLPayload = {
    hitl_type: "checkbox",
    title: "Select Topics",
    message: "Choose research topics",
    options: [
      { id: "t1", label: "AI Safety", description: "", selected: false },
      { id: "t2", label: "AI Ethics", description: "", selected: true },
      { id: "t3", label: "AI Policy", description: "", selected: false },
    ],
    report: null,
  };

  it("renders all options", () => {
    render(<HITLWidget payload={checkboxPayload} onResume={vi.fn()} />);
    expect(screen.getByText("AI Safety")).toBeInTheDocument();
    expect(screen.getByText("AI Ethics")).toBeInTheDocument();
    expect(screen.getByText("AI Policy")).toBeInTheDocument();
  });

  it("submit button shows selected count", () => {
    render(<HITLWidget payload={checkboxPayload} onResume={vi.fn()} />);
    // t2 is pre-selected
    expect(screen.getByText(/Continue with 1 selected/)).toBeInTheDocument();
  });

  it("calls onResume with selected IDs when submitted", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={checkboxPayload} onResume={onResume} />);

    // Click on "AI Safety" to select it
    await userEvent.click(screen.getByText("AI Safety"));

    // Click submit
    await userEvent.click(screen.getByText(/Continue with 2 selected/));

    expect(onResume).toHaveBeenCalledTimes(1);
    const calledWith = onResume.mock.calls[0][0] as string[];
    expect(calledWith).toContain("t1");
    expect(calledWith).toContain("t2");
  });

  it("submit disabled when none selected", () => {
    const noSelectionPayload: HITLPayload = {
      ...checkboxPayload,
      options: checkboxPayload.options.map((o) => ({ ...o, selected: false })),
    };
    render(<HITLWidget payload={noSelectionPayload} onResume={vi.fn()} />);
    const button = screen.getByText(/Continue with 0 selected/);
    expect(button).toBeDisabled();
  });
});

describe("YesNoWidget", () => {
  const yesNoPayload: HITLPayload = {
    hitl_type: "yes_no",
    title: "Confirm",
    message: "Proceed?",
    options: [],
    report: null,
  };

  it("renders two buttons", () => {
    render(<HITLWidget payload={yesNoPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("calls onResume with yes action", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={yesNoPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("Yes"));
    expect(onResume).toHaveBeenCalledWith({ action: "yes" });
  });

  it("calls onResume with no action", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={yesNoPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("No"));
    expect(onResume).toHaveBeenCalledWith({ action: "no" });
  });
});

describe("SelectWidget", () => {
  const selectPayload: HITLPayload = {
    hitl_type: "select",
    title: "Choose Option",
    message: "Pick one",
    options: [
      { id: "opt1", label: "Option A", description: "First", selected: false },
      { id: "opt2", label: "Option B", description: "Second", selected: false },
    ],
    report: null,
  };

  it("renders options", () => {
    render(<HITLWidget payload={selectPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Option A")).toBeInTheDocument();
    expect(screen.getByText("Option B")).toBeInTheDocument();
  });

  it("confirm button disabled initially", () => {
    render(<HITLWidget payload={selectPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Confirm")).toBeDisabled();
  });

  it("calls onResume with selected option", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={selectPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("Option A"));
    await userEvent.click(screen.getByText("Confirm"));
    expect(onResume).toHaveBeenCalledWith({ action: "opt1" });
  });
});

describe("TextWidget", () => {
  const textPayload: HITLPayload = {
    hitl_type: "text",
    title: "Enter Text",
    message: "Type your response",
    options: [],
    report: null,
  };

  it("submit disabled when empty", () => {
    render(<HITLWidget payload={textPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Submit")).toBeDisabled();
  });

  it("calls onResume with text value", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={textPayload} onResume={onResume} />);
    const textarea = screen.getByPlaceholderText("Type your response...");
    await userEvent.type(textarea, "My response");
    await userEvent.click(screen.getByText("Submit"));
    expect(onResume).toHaveBeenCalledWith({ text: "My response" });
  });
});

describe("ReviewWidget", () => {
  const reviewPayload: HITLPayload = {
    hitl_type: "review",
    title: "Review Report",
    message: "Review the report",
    options: [],
    report: mockReport,
  };

  it("renders report preview", () => {
    render(<HITLWidget payload={reviewPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Test Report")).toBeInTheDocument();
    expect(screen.getByText("A test summary about AI research")).toBeInTheDocument();
  });

  it("renders key findings", () => {
    render(<HITLWidget payload={reviewPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Key Findings")).toBeInTheDocument();
    expect(screen.getByText(/AI is advancing rapidly/)).toBeInTheDocument();
  });

  it("approve button calls onResume with approve action", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={reviewPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("Approve"));
    expect(onResume).toHaveBeenCalledWith({ action: "approve" });
  });

  it("redo button calls onResume with redo action", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={reviewPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("Redo"));
    expect(onResume).toHaveBeenCalledWith({ action: "redo" });
  });

  it("shows edit mode when Edit clicked", async () => {
    render(<HITLWidget payload={reviewPayload} onResume={vi.fn()} />);
    await userEvent.click(screen.getByText("Edit"));
    expect(screen.getByText("Edit Summary")).toBeInTheDocument();
    expect(screen.getByText("Save & Approve")).toBeInTheDocument();
  });
});

describe("ConfirmWidget", () => {
  const confirmPayload: HITLPayload = {
    hitl_type: "confirm",
    title: "Save Report",
    message: "Save to database?",
    options: [],
    report: mockReport,
  };

  it("renders report preview", () => {
    render(<HITLWidget payload={confirmPayload} onResume={vi.fn()} />);
    expect(screen.getByText("Test Report")).toBeInTheDocument();
  });

  it("save button calls onResume with save action", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={confirmPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("Save to Database"));
    expect(onResume).toHaveBeenCalledWith({ action: "save" });
  });

  it("cancel button calls onResume with cancel action", async () => {
    const onResume = vi.fn();
    render(<HITLWidget payload={confirmPayload} onResume={onResume} />);
    await userEvent.click(screen.getByText("Cancel"));
    expect(onResume).toHaveBeenCalledWith({ action: "cancel" });
  });
});
