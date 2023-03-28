\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TP}[10/10/2019 Stephane Weissbaum]

\LoadClass[a4paper,12pt,twoside,twocolumn]{article}

%===============================================
\RequirePackage{amsmath}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[colorlinks,bookmarks=false,linkcolor=black,urlcolor=blue, citecolor=black]{hyperref}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{units}
\RequirePackage{siunitx}
\RequirePackage{verbatim}
\RequirePackage{verbdef}
\RequirePackage{float}
\RequirePackage{multirow}
\RequirePackage{tabularx}
\RequirePackage{subcaption}
\RequirePackage{wrapfig}
\RequirePackage{tikz}
\RequirePackage{geometry}
\RequirePackage{enumitem}
\RequirePackage{balance}
\RequirePackage{lipsum}
\RequirePackage{stfloats}




%===============================================
\geometry{top=2.5cm, bottom=2cm, left=2cm, right=2cm}
\setlength\columnsep{2pc}





%\setcounter{secnumdepth}{0}

%Largeur line en-tête
\renewcommand{\headrulewidth}{0.1pt}

%\renewcommand\britishhyphenmins{33}
%\providehyphenmins{british}{33}


\def\hlinewd#1{%
\noalign{\ifnum0=`}\fi\hrule \@height #1 %
\futurelet\reserved@a\@xhline}

\def\institution#1{%
  \gdef\@institution{\medskip\noindent\strut{\itshape #1}}}
\def\assistant#1{%
  \gdef\@assistant{\medskip\noindent\strut{\itshape Assistant: #1}}}

\newbox\absbox
\def\abstract{\lrbox\absbox\minipage{.75\textwidth}%
  %\hrule\@height.1pt
  \medskip
  \normalfont\bfseries\small
  \leftskip\fill
  \rightskip-\fill
  \parfillskip\stretch2}
\def\endabstract{\endminipage\endlrbox}

\def\maketitle{{%
  \long\def\@makefntext##1{\parindent 1em\noindent
    \@textsuperscript{\normalfont\@thefnmark}##1}%
  \twocolumn[%
    \thispagestyle{plain}\label{FirstPage}%
    \normalfont\centering
    {\parbox[b]{.75\textwidth}{\centering
      {\Huge\@title\par}%
      \addvspace{17pt}% 
      \def\@makefnmark{\@textsuperscript{\normalfont\@thefnmark}}%
      {\scshape\@author\par}
       \addvspace{5pt}
    \small\@institution\hspace{1cm}\small\@assistant\par
     \vspace{-0.2cm}
     \today\par 
      \addvspace{15pt}
      \hrule\@height.2pt
     \usebox\absbox\par
     \addvspace\bigskipamount
     \small\itshape
     \addvspace\bigskipamount
  }}%
  \hrule\@height.2pt
  \vspace{20pt}%

  ]
}}




\def\section{%
  \@startsection
    {section}{1}{\z@}{-14\p@ \@plus -2\p@ \@minus -2\p@}%
    {6\p@ plus 2\p@}{\normalfont\normalsize\bfseries\MakeUppercase}}
\def\subsection{%
  \@startsection
    {subsection}{2}{\z@}{-14\p@ \@plus -2\p@ \@minus -2\p@}%
    {6\p@ plus 2\p@}{\normalfont\normalsize\bfseries}}
\def\subsubsection{%
  \@startsection
    {subsubsection}{3}{\z@}{-14\p@ \@plus -2\p@ \@minus-2\p@}%
    {1sp}{\normalfont\normalsize\itshape}}
    
    
    \captionsetup{labelsep=endash, justification=centering} %font=small,
    
    