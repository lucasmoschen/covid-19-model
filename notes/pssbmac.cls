\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pssbmac}

\LoadClass[10pt, a4paper, twoside]{article}
\setlength{\textwidth}{15cm}
\setlength{\textheight}{21.0cm}
\setlength{\topmargin}{0cm}
\setlength{\oddsidemargin}{1.65cm}
\setlength{\evensidemargin}{1.65cm}

\usepackage{amsthm}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}{Lemma}[section]
\newtheorem{proposition}{Proposition}[section]
\newtheorem{definition}{Definition}[section]
\newtheorem{remark}{Remark}[section]
\newtheorem{corollary}{Corollary}[section]
\newtheorem{teorema}{Teorema}[section]
\newtheorem{lema}{Lema}[section]
\newtheorem{prop}{Proposi\c{c}\~ao}[section]
\newtheorem{defi}{Defini\c{c}\~ao}[section]
\newtheorem{obs}{Observa\c{c}\~ao}[section]
\newtheorem{cor}{Corol\'ario}[section]

\renewenvironment{abstract}
{\begin{list}{}{%
  \setlength{\rightmargin}{0.5cm}
  \setlength{\leftmargin}{0.5cm}
  \small} 
  \item[] }
{\end{list}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\ps@headreport}
{ \renewcommand{\@oddhead}
  {\begin{minipage}{\textwidth}
%HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
     {\normalsize {\bf Proceeding Series of the Brazilian
Society of Computational and Applied Mathematics}}
     {\normalsize {\em }} 
%HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
     \rule{\textwidth}{0.6pt} 
   \end{minipage}
  }
  \renewcommand{\@oddfoot}{}
  \renewcommand{\@evenhead}{\@oddhead}
  \renewcommand{\@evenfoot}{\@oddfoot}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand\criartitulo{\par
  \begingroup
    \renewcommand\thefootnote{\@arabic\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
      
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@criartitulo

    \thispagestyle{headreport}\@thanks
     \endgroup
  \global\let\thanks\relax
  \global\let\criartitulo\relax
  \global\let\@criartitulo\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@criartitulo{%
  \newpage
  \null
  \vskip 0em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
  \end{center}%
    \vskip 1.5em%
    {\normalsize
      \lineskip .5em%
      \begin{minipage}[b]{\textwidth}
        \@author
      \end{minipage}\par}%
    \vskip 1em%
    %{\large \@date}%
  \par
  \vskip 1em}
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newcommand{\runningheads}{\markboth}
\pagestyle{myheadings}

\normalsize

\setcounter{page}{1}

\endinput
