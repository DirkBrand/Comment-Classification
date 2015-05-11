import wx

from RatingPrediction.main import runRegressionModelTest
import RatingPrediction.main
import wx.lib.scrolledpanel as scrolled


APP_EXIT = 1

class ExampleApp(wx.Frame):
    def __init__(self, *args, **kwargs):
        super(ExampleApp, self).__init__(*args, **kwargs) 
            
        self.InitUI()

        # Set window parameters
        self.SetSize((500, 300))
        self.SetTitle('Comment Ranker')
        self.Centre()
        self.Show(True)
    
    
    def InitUI(self):    

        # Create Menu
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        
        # Add quit button
        qmi = wx.MenuItem(fileMenu, APP_EXIT, '&Quit\tCtrl+Q')
        qmi.SetBitmap(wx.Bitmap('Images/exit.png'))
        fileMenu.AppendItem(qmi)
        
        # Bind Quit event
        self.Bind(wx.EVT_MENU, self.OnQuit, id=APP_EXIT)
        
        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)
        
        
        # The Main Panel
        mainPanel = wx.Panel(self)
        mainPanel.SetBackgroundColour('#4f5049')
        
        
        # Horizontal Sizer
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # The Button panel
        btnPan = wx.Panel(mainPanel)
        btnPan.SetBackgroundColour('#ededed')
        vboxl = wx.BoxSizer(wx.VERTICAL)
        radioPan = wx.Panel(btnPan)
        
        self.rb1 = wx.RadioButton(radioPan, label='Linear Regression', pos=(10, 10), 
            style=wx.RB_GROUP)
        self.rb2 = wx.RadioButton(radioPan, label='SVR', pos=(10, 30))
        self.rb3 = wx.RadioButton(radioPan, label='ELM', pos=(10, 50))
        vboxl.Add(radioPan, border=5)
                
        # Buttons
        classifyBtn = wx.Button(btnPan, label='Test Classifier', pos=(110, 90))
        loadBtn = wx.Button(btnPan, label='Load Data', pos=(10, 90))
        
        classifyBtn.Bind(wx.EVT_BUTTON, self.runClassifyerTest)
        
        btnPan.SetSizer(vboxl)
        
        
        
        hbox.Add(btnPan, 1, wx.EXPAND | wx.ALL, 10)
        
        # The text Panel
        textPnl = scrolled.ScrolledPanel(mainPanel, -1)
        textPnl.SetBackgroundColour('#ededed')
        
        vboxr = wx.BoxSizer(wx.VERTICAL)
        self.st1 = wx.StaticText(textPnl, label='Regression Stats\n=======================\n', style=wx.ALIGN_LEFT)
        vboxr.Add(self.st1,0, wx.ALL, border=10)
        textPnl.SetAutoLayout(1)
        textPnl.SetupScrolling()
        
        textPnl.SetSizer(vboxr)
        
        hbox.Add(textPnl, 1, wx.EXPAND | wx.ALL, 10)
        
        
        mainPanel.SetSizer(hbox)

        
    def runClassifyerTest(self,e):        
        if self.rb1.IsEnabled():
            self.appendToTextArea(runRegressionModelTest(1))
        elif self.rb2.IsEnabled():
            self.appendToTextArea(runRegressionModelTest(2))
        elif self.rb3.IsEnabled():
            self.appendToTextArea(runRegressionModelTest(3))
            
    
    def OnQuit(self, e):
        self.Close()
 
    def appendToTextArea(self, text):
        self.st1.SetLabel(self.st1.GetLabel() + text)
        
# Instantiate and run

ex = wx.App()
ExampleApp(None)
ex.MainLoop()  
