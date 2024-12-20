package game;

import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        // 确保在EDT线程中运行
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Space Shooter");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);
            frame.add(new Game());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
